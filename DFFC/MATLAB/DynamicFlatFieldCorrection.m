%Dynamic intensity normalization using eigen flat fields in X-ray imaging
%--------------------------------------------------------------------------
%
% Script: Computes the conventional and dynamic flat field corrected
% projections of a computed tomography dataset.
%
% Input:
% Dark fields, flat fields and projection images in .tif format.
%
% Output:
% Dynamic flat field corrected projections in map
% 'outDIRDFFC'. Conventional flat field corrected projtions in
% map 'outDIRFFC'.
%
%More information: V.Van Nieuwenhove, J. De Beenhouwer, F. De Carlo, L.
%Mancini, F. Marone, and J. Sijbers, "Dynamic intensity normalization using
%eigen flat fields in X-ray imaging", Optics Express, 2015
%
%--------------------------------------------------------------------------
%Vincent Van Nieuwenhove                                        13/10/2015
%vincent.vannieuwenhove@uantwerpen.be
%iMinds-vision lab
%University of Antwerp
%% parameters

%directories
% Directory with raw dark fields, flat fields and projections in .tif format
readDIR=    '../../input/duplicate_testing/real_0/';
% Directory where the DYNAMIC flat field corrected projections are saved
outDIRDFFC= '../../output/DFFC/MATLAB';

addpath('/Users/sampieters/Desktop/informatica/master/Research Project/CodeDynamicFlatFieldCorrection/BM3D')              % directory with files of BM3D filter

%file names
prefixProj=         'dbeer_5_5_';   % prefix of the original projections
outPrefixDFFC=      'DFFC';         % prefix of the DYNAMIC flat field corrected projections
prefixFlat=         'dbeer_5_5_';   % prefix of the flat fields
prefixDark=         'dbeer_5_5_';   % prefix of the dark fields
numType=            '%04d';         % number type used in image names
fileFormat=         '.tif';         % image format

nrDark=             20;             % number of dark fields
firstDark=          1;              % image number of first dark field
nrWhitePrior=       300;            % number of white (flat) fields BEFORE acquiring the projections
firstWhitePrior=    21;             % image number of first prior flat field
nrWhitePost=        300;            % number of white (flat) fields AFTER acquiring the projections
firstWhitePost=     572;            % image number of first post flat field
nrProj=             50;        	    % number of acquired projections
firstProj=          321;            % image number of first projection

% options output images
scaleOutputImages=  [0 2];          %output images are scaled between these values

% algorithm parameters
downsample=         2;              % amount of downsampling during dynamic flat field estimation (integer between 1 and 20)
nrPArepetions=      10;             % number of parallel analysis repetions


%% load dark and white fields

mkdir(outDIRDFFC)
%mkdir(outDIRFFC)

nrImage=firstProj:firstProj-1+nrProj;
display('load dark and flat fields:')
tmp=imread([readDIR prefixProj num2str(1,numType) fileFormat]);
dims=size(tmp);

%load dark fields
display('Load dark fields ...')
dark=zeros([dims(1) dims(2) nrDark]);
for ii=firstDark:firstDark+nrDark-1
    dark(:,:,ii)=double(imread([readDIR prefixDark num2str(ii,numType) fileFormat]));
end
meanDarkfield = mean(dark,3);

%load white fields
whiteVec=zeros([dims(1)*dims(2) nrWhitePrior+nrWhitePost]);

display('Load white fields ...')
k=0;
for ii=firstWhitePrior:firstWhitePrior-1+nrWhitePrior
    k=k+1;
    tmp2=double(imread([readDIR prefixFlat num2str(ii,numType) fileFormat]));
    tmp=tmp2-meanDarkfield;
    whiteVec(:,k)=tmp(:)-meanDarkfield(:);
end
for ii=firstWhitePost:firstWhitePost-1+nrWhitePost
    k=k+1;
    tmp=double(imread([readDIR prefixFlat num2str(ii,numType) fileFormat]))-meanDarkfield;
    whiteVec(:,k)=tmp(:)-meanDarkfield(:);
end
mn = mean(whiteVec,2);

% substract mean flat field
[M,N] = size(whiteVec);
hulp = repmat(mn, 1,N);
Data = whiteVec - repmat(mn,1,N);
clear whiteVec dark

%% calculate Eigen Flat fields
% Parallel Analysis
display('Parallel Analysis:')
[V1, D1, nrEigenflatfields]=parallelAnalysis(Data,nrPArepetions);
display([int2str(nrEigenflatfields) ' eigen flat fields selected.'])

%calculation eigen flat fields
eig0 = reshape(mn,dims);
EigenFlatfields(:,:,1) = eig0;
for ii=1:nrEigenflatfields
    t = V1(:,N-ii+1);
    k = Data*V1(:,N-ii+1);

    tmp = reshape(Data*V1(:,N-ii+1),dims);
    EigenFlatfields(:,:,ii+1) = tmp;
end

clear Data

%% Filter Eigen flat fields
display('Filter eigen flat fields ...')
filteredEigenFlatfields=zeros(dims(1),dims(2),1+nrEigenflatfields);

for ii=2:1+nrEigenflatfields
    display(['filter eigen flat field ' int2str(ii-1)])
    % tmp is the Eigenflatfields normalized between 0 and 1
    tmp=(EigenFlatfields(:,:,ii)-min(min(EigenFlatfields(:,:,ii))))/(max(max(EigenFlatfields(:,:,ii)))-min(min(EigenFlatfields(:,:,ii))));
    imwrite(tmp,[outDIRDFFC '/eigenflatfields/eigenflatfield_' num2str(ii-1,numType) fileFormat]);
    [~,tmp2]=BM3D(1,tmp);
    filteredEigenFlatfields(:,:,ii)=(tmp2*(max(max(EigenFlatfields(:,:,ii)))-min(min(EigenFlatfields(:,:,ii)))))+min(min(EigenFlatfields(:,:,ii)));
end

%% estimate abundance of weights in projections
meanVector=zeros(1,length(nrImage));
xArray=zeros(nrEigenflatfields,length(nrImage));
for ii=1:length(nrImage)
    display(['estimation projection ' int2str(ii) '/' int2str(length(nrImage)) '...'])
    %load projection
    projection=double(imread([readDIR prefixProj num2str(nrImage(ii),numType) fileFormat]));
    tmp=(squeeze(projection)-meanDarkfield)./(EigenFlatfields(:,:,1));
    meanVector(ii)=mean(tmp(:));
    
    %estimate weights for a single projection
    x=condTVmean(projection,EigenFlatfields(:,:,1),filteredEigenFlatfields(:,:,2:(1+nrEigenflatfields)),meanDarkfield,zeros(1,nrEigenflatfields),downsample);
    xArray(:,ii)=x;
    
    %dynamic flat field correction
    FFeff=zeros(size(meanDarkfield));
    for  j=1:nrEigenflatfields
        FFeff=FFeff+x(j)*filteredEigenFlatfields(:,:,j+1);
    end
    
    tmp=(squeeze(projection)-meanDarkfield)./(EigenFlatfields(:,:,1)+FFeff);
    tmp=tmp/mean(tmp(:))*meanVector(ii);
    tmp(tmp<0)=0;
    tmp=-log(tmp);
    tmp(isinf(tmp))=10^5;
    tmp=(tmp-scaleOutputImages(1))/(scaleOutputImages(2)-scaleOutputImages(1));
    tmp=uint16((2^16-1)*tmp);
    imwrite(tmp,[outDIRDFFC outPrefixDFFC num2str(nrImage(ii),numType) fileFormat]);
end

save([outDIRDFFC '\' 'parameters.mat'], 'xArray')