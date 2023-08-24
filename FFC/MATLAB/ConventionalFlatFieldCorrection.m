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
tic
%directories
% Directory with raw dark fields, flat fields and projections in .tif format
readDIR= './XrayDenoising/input/benchmark/noisy/';       
% Directory where the CONVENTIONAL flat field corrected projections are saved
outDIRFFC=  './XrayDenoising/output/FFC/MATLAB/';

%file names
prefixProj=         'dbeer_5_5_';   % prefix of the original projections
outPrefixFFC=       'FFC_';         % prefix of the CONVENTIONAL flat field corrected projections
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
nrProj=             250;        	% number of acquired projections
firstProj=          321;            % image number of first projection

% options output images
scaleOutputImages=  [0 1];          %output images are scaled between these values

%% load dark and white fields
mkdir(outDIRFFC)

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
    tmp=double(imread([readDIR prefixFlat num2str(ii,numType) fileFormat]))-meanDarkfield;
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
eig0 = reshape(mn,dims);
clear whiteVec dark Data

%% estimate abundance of weights in projections
meanVector=zeros(1,length(nrImage));
for ii=1:length(nrImage)
    display(['conventional FFC: ' int2str(ii) '/' int2str(length(nrImage)) '...'])
    %load projection
    projection=double(imread([readDIR prefixProj num2str(nrImage(ii),numType) fileFormat]));
    
    tmp=(squeeze(projection)-meanDarkfield)./ eig0;
    meanVector(ii)=mean(tmp(:));
    
    tmp(tmp<0)=0;
    tmp=-log(tmp);
    tmp(isinf(tmp))=10^5;
    tmp=(tmp-scaleOutputImages(1))/(scaleOutputImages(2)-scaleOutputImages(1));
    tmp=uint16((2^16-1)*tmp);
    imwrite(tmp,[outDIRFFC outPrefixFFC num2str(nrImage(ii),numType) fileFormat]);
end
toc