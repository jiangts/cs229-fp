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
sample_length = 1;
feature_extraction_per_sec = 100;
num_features = 30; % alpha - 30, num - 30, special - 30
file_sample_length = 44101;
files_to_go_through = 1;

%words = [26, 10, 14];
words = [26, 10, 12];
%symbol_list = cellstr(['exclamation'; 'at         '; 'hash       '; 'dollar     ';
%               'percent    '; 'caret      '; 'and        '; 'star       '; 
%               'underscore '; 'dash       '; 'comma      '; 'period     '; 
%               'question   '; 'tilde      ']);
symbol_list = cellstr(['at         '; 'hash       '; 'dollar     ';
               'percent    '; 'caret      '; 'and        '; 'star       '; 
               'dash       '; 'comma      '; 'period     '; 
               'question   '; 'tilde      ']);

feature_points_alpha = [];
labels_alpha = [];
feature_points_num = [];
labels_num = [];
feature_points_special = [];
labels_special = [];
temp_feature_points = [];
           
# for type=3:3
for type=1:1
    num_words_in_category = words(type);
    for i=1:num_words_in_category
        disp(i);
        for fff=1:files_to_go_through
            word_name = 0;
            if(type==1)
                word_name = char(96+i);
            elseif(type==2)
                word_name = char(47+i);
            else
                word_name = char(symbol_list(i));
            end
            if fff==1
                files = dir(strcat('slicedup/',word_name,'_*.wav'));
            else
                files = dir(strcat('slicedup_dirty/',word_name,'_*.wav'));
            end
            %files = dir(strcat('preprocessed/processed_',word_name,'_*.wav'));
            if i==1
                temp_feature_points = [];
                temp_feature_points_non_pca = [];
                temp_labels = [];
            end

            file_num = 1;
            for f=files'
                %disp(f.name);
                if fff==1
                    [raw_speech, Fs] = wavread(strcat('slicedup/',f.name));
                else
                    [raw_speech, Fs] = wavread(strcat('slicedup_dirty/',f.name));
                end
                %[raw_speech, Fs] = audioread(strcat('preprocessed/',f.name));
                if length(raw_speech)~=file_sample_length
                    disp('File wrong size');
                    disp(f.name);
                    disp(length(raw_speech));
                    continue;
                end
                raw_speech = raw_speech/std(raw_speech);
                %raw_speech = raw_speech(round(length(raw_speech)*(1/2-sample_length/2)+1):round(length(raw_speech)*(1/2+sample_length/2)-1));
                [ceps,freqresp,fb,fbrecon,freqrecon] = mfcc(raw_speech, Fs, feature_extraction_per_sec);
                
                feature_vec = [];
                feature_vec2 = [];
                feature_vec_non_pca = [];
                % feature point #1 - raw MFCC, no DC term
                ceps_no_dc = ceps(remove_0+1:13-remove_top,:);
                feature_vec = reshape(ceps_no_dc,size(ceps_no_dc,1)*size(ceps_no_dc,2),1);

                iii = 1;
                ceps_block = [];
                block_sz = 10;
                for ii=1:block_sz:length(ceps)-block_sz
                    ceps_block = [ceps_block mean(ceps(:,ii:ii+block_sz),2)];
                    iii = iii+1;
                end
                % feature point #2 - differentiate the MFCC terms
%                 cep_power_diff = diff(ceps_block,1,2);
%                 temp = reshape(cep_power_diff,size(cep_power_diff,1)*size(cep_power_diff,2),1);
%                 feature_vec = [feature_vec; temp];
                
                % feature point #3 - differentiate the MFCC DC term only
%                 cep_power_diff = diff(ceps_block(1,:));
%                 feature_vec_non_pca = [feature_vec_non_pca; cep_power_diff'];
                
                % feature point #4 - MFCC DC term 2nd derivative
%                 cep_power_diff = diff(diff(ceps_block(1,:)));
%                 feature_vec_non_pca = [feature_vec_non_pca; cep_power_diff'];

                % feature point #5 - Signal Power Analysis
%                 num_shift = 128;
%                 sec_sub = round(Fs/128);
%                 shift_amount = round(length(raw_speech)/num_shift);
%                 sound_sample_sq = raw_speech.^2;
%                 start_index = 1;
%                 section_power = [];
%                 while start_index+sec_sub<length(sound_sample_sq)
%                     section_power = [section_power sum(sound_sample_sq(start_index:start_index+sec_sub))];
%                     start_index = start_index + shift_amount;
%                 end
%                 end_samples = [section_power(1:20)];
%                 end_samples(end_samples>mean(end_samples)+std(end_samples))=[];
%                 %noise_avg = sum(end_samples)/length(end_samples);
%                 noise_avg = 100;
%                 threshold = 1;
%                 % half point method
%                 half_point = round(length(section_power)/2);
%                 prev = section_power(half_point); 
%                 for ii=half_point:-1:1
%             %         if prev<section_power(i) && section_power(i)<noise_avg*2
%             %             break;
%             %         end
%             %         prev = section_power(i);
%                     if section_power(ii)<threshold*noise_avg
%                         break
%                     end
%                 end
%                 starting = ii-1;
%                 prev = section_power(half_point); 
%                 for ii=round(half_point):length(section_power)
%             %         if prev<section_power(i) && section_power(i)<noise_avg*2
%             %             break;
%             %         end
%             %         prev = section_power(i);
%                     if section_power(ii)<threshold*noise_avg
%                         break
%                     end
%                 end
%                 ending = ii-1;
%                 feature_vec_non_pca = [feature_vec_non_pca; ending-starting];

                temp_feature_points = [temp_feature_points; feature_vec'];
                %temp_feature_points2 = [temp_feature_points2; feature_vec2'];
                temp_feature_points_non_pca = [temp_feature_points_non_pca; feature_vec_non_pca'];
                temp_labels = [temp_labels i];
                file_num = file_num + 1;
            end
        end
    end
    
    % PCA analysis
    if size(temp_feature_points,1)>num_features && size(temp_feature_points,2)>num_features
         % set the average to be 0
         temp_feature_points = bsxfun(@minus,temp_feature_points,mean(temp_feature_points,1));
         % set the st. dev. to be 1
         std_dev = std(temp_feature_points,1);
         std_dev = std_dev + (std_dev==0)*1;
         temp_feature_points = bsxfun(@rdivide,temp_feature_points,std_dev);
 
         C = cov(temp_feature_points);
         [U,S,V] = svd(C);
         pca_coeff = V(:,1:num_features);

        %pca_coeff = pca(temp_feature_points);
        
        
        temp_feature_points_shrunk = temp_feature_points*pca_coeff(:,1:num_features);
    else
        temp_feature_points_shrunk = temp_feature_points;
    end
    temp_feature_points_shrunk = [temp_feature_points_shrunk temp_feature_points_non_pca];
    
    if(type==1)
        feature_points_alpha = temp_feature_points_shrunk;
        labels_alpha = temp_labels;
    elseif(type==2)
        feature_points_num = temp_feature_points_shrunk;
        labels_num = temp_labels;
    else
        feature_points_special = temp_feature_points_shrunk;
        labels_special = temp_labels;
    end
end

%save('alpha_feature.mat','feature_points_alpha');
%save('alpha_label.mat','labels_alpha');
%save('num_feature.mat','feature_points_num');
%save('num_label.mat','labels_num');
save -6 alpha_feature.mat feature_points_alpha;
save -6 alpha_label.mat labels_alpha;
save -6 num_feature.mat feature_points_num;
save -6 num_label.mat labels_num;
save -6 special_feature.mat feature_points_special;
save -6 special_label.mat labels_special;

%[status, result] = system('python test_2_num.py');
%disp(result);
test
