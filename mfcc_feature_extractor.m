% mfcc_feature_extractor.m
% CS229 Project
% Yuki Inoue
% 
% Function:
% Takes 1 second recordings of various word types, and runs Mel-Frequency
% Ceptrum Coefficient Transform (uses mfcc.m, developed by Malcolm Slaney).
% Stores the features stored in 3 vectors, prefixed with feature_points_
% 
% The audio file is taken from /slicedup, and the lower 13 ceptrums for
% mfcc is taken as the feature vectors, with (feature_extraction_per_sec-1) samples per file.
% Therefore, (feature_extraction_per_sec-1)*13 feature points are extracted
% from each file.
%
% The dimension of the feature_points_ vectors are (number of word types in a
% category) x (feature_extraction_per_sec-1)*13 x (number of sample files
% available in the /slicedup directory)
%
% Usage:
% feature_extraction_per_sec sets the number of times the script samples
% the audio file (i.e. the script divides the audio input into
% feature_extraction_per_sec sections and runs mfcc on them.)
%
% Adjust type in the first for loop to choose which feature points to
% collect (type=1: alphabets, type=2: numbers, type=3: special characters)
% 

clear all;
remove_0 = 1;
remove_top = 0;
feature_extraction_per_sec = 150;
num_feature_points_per_sec = (13-remove_0-remove_top)*(feature_extraction_per_sec-1);
num_total_features = num_feature_points_per_sec + feature_extraction_per_sec-2;
num_features = 50;
words = [26, 10, 14];
symbol_list = cellstr(['exclamation'; 'at         '; 'hash       '; 'dollar     ';
               'percent    '; 'caret      '; 'and        '; 'star       '; 
               'underscore '; 'dash       '; 'comma      '; 'period     '; 
               'question   '; 'tilde      ']);

feature_points_alpha = [];
feature_points_num = [];
feature_points_special = [];
temp_feature_points = [];
           
for type=1:1
    num_words_in_category = words(type);
    for i=1:num_words_in_category
        disp(i);
        word_name = 0;
        if(type==1)
            word_name = char(96+i);
        elseif(type==2)
            word_name = char(47+i);
        else
            word_name = char(symbol_list(i));
        end
        files = dir(strcat('slicedup/',word_name,'_*.wav'));
        %files = dir(strcat('preprocessed/processed_',word_name,'_*.wav'));
        if i==1
            temp_feature_points = zeros(num_words_in_category,length(files),num_total_features);
            %temp_feature_points = [];
        end

        file_num = 1;
        for f=files'
            %disp(f.name);
            [raw_speech, Fs] = audioread(strcat('slicedup/',f.name));
            %[raw_speech, Fs] = audioread(strcat('preprocessed/',f.name));
            raw_speech = raw_speech/std(raw_speech);
            [ceps,freqresp,fb,fbrecon,freqrecon] = mfcc(raw_speech, Fs, feature_extraction_per_sec);
            feature_vec = reshape(ceps(remove_0+1:13-remove_top,:),(size(ceps,1)-remove_0-remove_top)*size(ceps,2),1);
            feature_vec = [feature_vec; zeros(num_feature_points_per_sec-length(feature_vec),1)];
            
            cep_power_diff = diff(ceps(1,:));
            feature_vec = [feature_vec; cep_power_diff'];
            feature_vec = [feature_vec; zeros(num_total_features-length(feature_vec),1)];
            
            temp_feature_points(i,file_num,:) = feature_vec;
            file_num = file_num + 1;
        end
    end
    feature_matrix = reshape(temp_feature_points,num_words_in_category*(file_num-1),num_total_features);
    std_dev = std(feature_matrix,1);
    avgs = mean(feature_matrix,1);
    for i=1:num_feature_points_per_sec
        temp_feature_points(:,:,i) = temp_feature_points(:,:,i)-avgs(i);
        temp_feature_points(:,:,i) = temp_feature_points(:,:,i)/std_dev(i);
    end
    % PCA analysis
    C = cov(feature_matrix);
    [V,D] = eig(C);
    principal_vectors = V(:,end-num_features+1:end);
    temp_feature_points_shrunk = zeros(num_words_in_category,length(files),num_features);
    for i=1:num_words_in_category
        for j=1:length(files)
            temp_feature_points_shrunk(i,j,:) = reshape(temp_feature_points(i,j,:),1,num_total_features)*principal_vectors;
        end
    end
    
    if(type==1)
        feature_points_alpha = temp_feature_points_shrunk;
    elseif(type==2)
        feature_points_num = temp_feature_points_shrunk;
    else
        feature_points_special = temp_feature_points_shrunk;
    end
end

save('alpha_feature.mat','feature_points_alpha');
save('num_feature.mat','feature_points_num');

[status, result] = system('python test.py');
disp(result);