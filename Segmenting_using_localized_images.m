clc; % Clear command window.
clearvars; % Get rid of variables from prior run of this m-file.
%% Process Labels
% processing train labels
% set Train_labels path
fid = fopen('G:\Uppsala Docs\period3\CAIA-II\project\p_data\v_p_data\Labels_train.txt');
% read labels
data = textscan(fid,'%s');true_labels = data{1,1};
% close fopen
fclose(fid);
% empty variable to store each characters
train_labels = blanks(length(true_labels)*7);
% loop to segment labels into characters 
for y=1:length(true_labels)
   % split string into characters
   str_split = split(true_labels(y),"");
   % remove blank characters
   str_split(1) = [];str_split(8) = [];
   % appending to array
   if y==1
      train_labels = char(str_split);
   else
      train_labels = char(train_labels,char(str_split));
   end
end

% processing test labels
% set Test_labels path
fid = fopen('G:\Uppsala Docs\period3\CAIA-II\project\p_data\v_p_data\Labels_test.txt');
% read labels
data = textscan(fid,'%s');true_labels = data{1,1};
% close fopen
fclose(fid);
% empty variable to store each characters
test_labels = blanks(length(true_labels)*7);
% loop to segment labels into characters 
for y=1:length(true_labels)
   % split string into characters
   str_split = split(true_labels(y),"");
   % remove blank characters
   str_split(1) = [];str_split(8) = [];
   % appending to array
   if y==1
      test_labels = char(str_split);
   else
      test_labels = char(test_labels,char(str_split));
   end
end

%% Reading all the images from directory and Removing noise from the png images

% intialised variables
resized_numberplate_size = [60 240];
resized_character_size = [30 20];
list_segmented_characters = {};
list_number_plates = {};
n = 1;
p = 1;

% additional intialised variables
temp = {};
categories = ["crop_m1","crop_m2","crop_m3","crop_m4"];
folder = 'G:\Uppsala Docs\period3\CAIA-II\project\p_data\v_p_data\';

for m=1:length(categories)
    temp_fold = folder + categories(m);
    filePattern = fullfile(temp_fold, '*.png');
    srcFiles = dir(filePattern);
    numFiles = length(srcFiles);
for y = 1:numFiles
    fileName = temp_fold + '/' + srcFiles(y).name;
    image = imread(fileName);
    % preprocessing image 1
    image_clean = preprocessing_image1(image,resized_numberplate_size);
    % blob analysis
    [object_stats,character_image,character_stats] = blob_analysis(image_clean);
    % check if their are 7 character
    if length(character_stats) ~= 7
        % preprocessing image 2
        image_clean = preprocessing_image2(image,resized_numberplate_size);
        % blob analysis
        [object_stats,character_image,character_stats] = blob_analysis(image_clean);
    end
    
    for j = 1:size(character_stats,1)
        
        % character segmentation
        bounding_box = character_stats(j).BoundingBox;
        segemented_character = imcrop(character_image, bounding_box);
        resized_segemented_character = imresize(segemented_character...
            ,resized_character_size);
        
        % appending character to features array
        if m ~= 4
            train_features(n,:) = extractHOGFeatures(...
                resized_segemented_character,'CellSize',[3 3]);
            n=n+1;
        else
            test_features(p,:) = extractHOGFeatures(...
                resized_segemented_character,'CellSize',[3 3]);
            p=p+1;
        end
            
    end
    clc;
end
end

%% Train the model using SVM classifier
model = fitcecoc(train_features,train_labels,'Coding','onevsone','Learners',...
    'svm','ObservationsIn','rows','Verbose',2);
%save model.mat

%% Predict the label
predictedLabel = predict(model,test_features);
Accuracy = (sum(predictedLabel == test_labels)/length(test_labels));

%% Accuracy Based on each Number Plate
cnt=0;
flag=0;
for i=1:length(test_labels)
    if predictedLabel(i) ~= test_labels(i)
       flag = 1;
    else 
       if mod(i,7) == 0 
          if flag ~= 1
             cnt = cnt+1;
          else
             flag = 0; 
          end
       end
    end
end

Actually_Accuracy = cnt/(length(test_labels)/7);
%%
plotconfusion(test_labels,predictedLabel)

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
    index1 = ([object_stats.Area] >= 70 & [object_stats.Area] <= 350);
    index2 = ([object_stats.EquivDiameter] >= 9 & [object_stats.EquivDiameter] <= 25);
    index3 = ([object_stats.Solidity] <= 0.82);
    indexes = find(index1 & index2 & index3);
    character_image = ismember(image_labeled,indexes);
    character_stats = regionprops(character_image,'all');
end