clc; % Clear command window.
clearvars; % Get rid of variables from prior run of this m-file.
%% Reading all the images from directory and Removing noise from the png images

FilesPath = 'G:\Uppsala Docs\period3\CAIA-II\project\data\Data_Set_Bazak\040603\';
Files = dir(append(FilesPath,'*.jpg'));
numFiles = length(Files);


%% preprocessing image

for i = 8:8 %numFiles
    % reading image
    image = imread(append(FilesPath,Files(i).name));
    %image = imread('G:\Uppsala Docs\period3\CAIA-II\project\data\saved_images\03.jpg');
    % resizing image
    image_resize_1 = imresize(image, [480 NaN]);
    % converting to gray scale
    image_gray = rgb2gray(image_resize_1);
    % converting to binary
    image_binarize = imbinarize(image_gray);
    % generating edges
    image_edge = edge(imgaussfilt(image_gray,1.8),'sobel');

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
    % reducing the image size
    image_resize_2 = imresize(image_cropped, [60 240]);
    
    % opening to combine images at the edges and also to remove noise
    %imag_open = imopen(image_resize_2, strel('rectangle', [4 4]));
    % inverse
    %image_after_localisation = ~imag_open;
    
    %% display output
    figure(1)
    subplot(221),imshow(image_gray);
    subplot(222),imshow(image_binarize);
    subplot(223),imshow(image_eroded);
    subplot(224),imshow(image_resize_2);
    %pause(1);
end
    
%%
%figure(1),imshow(image);title('RGB Image')
%figure(2),imshow(image_binarize);title('Binary Image')
%figure(3),imshow(image_edge);title('Sobel Edge Image')
%figure(4),imshow(image_dilated);title('Dilating Edge Image')
%figure(5),imshow(image_holes_filled);title('Filling Image Holes')
%figure(6),imshow(image_eroded);title('Eroding Image')
%figure(7),imshow(image_dilated_2);title('Dilating Number Plate Boundary')
%figure(8),imshow(image_cropped);title('Cropping ROI')
%figure(9),imshow(image_resize_2);
%figure(10),imshow(image_after_localisation);
%%