clc; % Clear command window.
clearvars; % Get rid of variables from prior run of this m-file.
%% Reading all the images from directory and Removing noise from the png images

FilesPath = 'G:\Uppsala Docs\period3\CAIA-II\project\data\Data_Set_Bazak\040603\';
Files = dir(append(FilesPath,'*.jpg'));
numFiles = length(Files);

% intialised variables
resized_numberplate_size = [60 240];
resized_character_size = [30 20];
list_segmented_characters = {};
n = 1;
load model.mat

%% preprocessing image

for i = 8:8 %numFiles
    % reading image
    image = imread(append(FilesPath,Files(i).name));
    % resizing image
    image_resize_1 = imresize(image, [480 NaN]);
    % converting to gray scale
    image_gray = rgb2gray(image_resize_1);
    % converting to binary
    image_binarize = imbinarize(image_gray);
    % generating edges
    image_edge = edge(imgaussfilt(image_gray,1.8),'sobel');
    %----------------------%
    figure(1)
    subplot(221),imshow(image);
    %% segmenting the region of number plate

    % dilating edges
    image_dilated = imdilate(image_edge, strel('diamond', 2));
    % filling the region with concealed edges
    image_holes_filled = imfill(image_dilated, 'holes');
    % remove the edges smaller than 10
    image_eroded = imerode(image_holes_filled, strel('diamond', 10));
    % dilate to increase the boundary region
    image_dilated_2 = imdilate(image_eroded, strel('diamond', 2));
    % get properties of remaning objects 
    object_stats = regionprops(image_dilated_2,'all');
    % intialise place holder for no of objects and max area
    object_area = object_stats.Area;
    no_of_objects = numel(object_stats);
    max_area = object_area;
    object_boundingbox = object_stats.BoundingBox;
    
    % finding the object with largest area which is number plate
    for i=1:no_of_objects
        if max_area < object_stats(i).Area
           max_area = object_stats(i).Area;
           object_boundingbox = object_stats(i).BoundingBox;
        end
    end
    %% segmenting the number plate
    
    % cropping image
    image_cropped = imcrop(image, object_boundingbox);
    %----------------------------%
    subplot(222),imshow(image_cropped);
    %% segmenting individual Characters
    
    % preprocessing image 1
    image_clean = preprocessing_image1(image_cropped,resized_numberplate_size);
    % blob analysis
    [object_stats,character_image,character_stats] = blob_analysis(image_clean);
    % check if their are 7 character
    if length(character_stats) ~= 7
        % preprocessing image 2
        image_clean = preprocessing_image2(image_cropped,resized_numberplate_size);
        % blob analysis
        [object_stats,character_image,character_stats] = blob_analysis(image_clean);
    end
    subplot(223),imshow(image_clean);
    for j = 1:size(character_stats,1)
        
        % character segmentation
        bounding_box = character_stats(j).BoundingBox;
        segemented_character = imcrop(character_image, bounding_box);
        resized_segemented_character = imresize(segemented_character...
            ,resized_character_size);
        
        list_segmented_characters{n} = mat2cell(resized_segemented_character...
            ,resized_character_size(1),resized_character_size(2));
        
        n = n+1;
    end

    %% 

    for x = 1:length(list_segmented_characters)
        figure(2);
        subplot(1,length(list_segmented_characters),x);
        imshow(cell2mat(list_segmented_characters{1,x}));
        features(x,:) = extractHOGFeatures(cell2mat(list_segmented_characters{1,x}),'CellSize',[3 3]);
    end
    pause(1);
    
    % predict labels
    tic;
    pred_Labels = predict(model,features);
    toc;
    
end

%% Functions
% preprocessing image 1
function image_clean = preprocessing_image1(image,resized_numberplate_size)
    image_gray = rgb2gray(image);
    image_noise_removed = medfilt2(image_gray,[3 3]);
    image_sharpen = imsharpen(image_noise_removed,'Radius',2,'Amount',1);
    image_resized = imresize(image_sharpen,resized_numberplate_size);
    T = graythresh(image_resized);
    image_binary = imbinarize(image_resized,T);
    se = strel('disk',1);
    image_morph_close = imclose(image_binary,se);
    image_labeled = bwlabel(~image_morph_close,8);
    image_clean = imclearborder(image_labeled,8);
end
% preprocessing image 2
function image_clean = preprocessing_image2(image,resized_numberplate_size)
    image_gray = rgb2gray(image);
    image_noise_removed = medfilt2(image_gray);
    image_sharpen = imsharpen(image_noise_removed,'Radius',2,'Amount',1);
    image_resized = histeq(imresize(image_sharpen,resized_numberplate_size));
    T = graythresh(image_resized);
    image_binary = imbinarize(image_resized,T);
    se = strel('rectangle',[2 2]);
    image_morph_close = imclose(image_binary,se);
    image_labeled = bwlabel(~image_morph_close,8);
    image_clean = imclearborder(image_labeled,8);
end
% blob analysis
function [object_stats,character_image,character_stats] = blob_analysis(image_labeled)
    object_stats = regionprops(image_labeled,'all');
    index1 = ([object_stats.Area] >= 180 & [object_stats.Area] <= 500); % 70 350
    index2 = ([object_stats.EquivDiameter] >= 9 & [object_stats.EquivDiameter] <= 25);
    index3 = ([object_stats.Solidity] <= 0.82);
    indexes = find(index1 & index2 & index3);
    character_image = ismember(image_labeled,indexes);
    character_stats = regionprops(character_image,'all');
end