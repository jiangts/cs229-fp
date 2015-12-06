% sound_preprocess.m
% CS229 Project
% Yuki Inoue
% 
% Function:
%
% Usage:
% 

clear all;
files = dir('slicedup/*a_alpha*.wav');
file_num = 1;
recording_length = 44101;
plotting = 1;
for f=files'
    disp(f.name);
    [raw_speech, Fs] = audioread(strcat('slicedup/',f.name));
    raw_speech = raw_speech/std(raw_speech);
    if plotting
        figure(1)
        plot(raw_speech)
    end
    
    % spectral subtraction
    if(length(raw_speech)~=recording_length)
        continue;
    end    
    
    % figure out the no voice part
    num_shift = 128;
    sec_sub = round(Fs/128);
    shift_amount = round(length(raw_speech)/num_shift);
    sound_sample_sq = raw_speech.^2;
    start_index = 1;
    section_power = [];
    while start_index+sec_sub<length(sound_sample_sq)
        section_power = [section_power sum(sound_sample_sq(start_index:start_index+sec_sub))];
        start_index = start_index + shift_amount;
    end
    end_samples = [section_power(1:20)];
    end_samples(end_samples>mean(end_samples)+std(end_samples))=[];
    %noise_avg = sum(end_samples)/length(end_samples);
    noise_avg = 100;
    threshold = 1;
    % half point method
    half_point = round(length(section_power)/2);
    prev = section_power(half_point); 
    for i=half_point:-1:1
%         if prev<section_power(i) && section_power(i)<noise_avg*2
%             break;
%         end
%         prev = section_power(i);
        if section_power(i)<threshold*noise_avg
            break
        end
    end
    starting = i-1;
    prev = section_power(half_point); 
    for i=round(half_point):length(section_power)
%         if prev<section_power(i) && section_power(i)<noise_avg*2
%             break;
%         end
%         prev = section_power(i);
        if section_power(i)<threshold*noise_avg
            break
        end
    end
    ending = i-1;

    if plotting
        figure(4)
        plot(section_power)
    end
% %     end-to-end method
%     for i=1:length(section_power)
%         hill = 1;
%         for m=0:7
%             if section_power(i+m)<noise_avg*threshold
%                 hill = 0;
%                 break;
%             end
%         end
%         if hill==1
%             break;
%         end
%     end
%     starting = i;
%     for i=length(section_power):-1:1
%         hill = 1;
%         for m=0:7
%             if section_power(i-m)<noise_avg*threshold
%                 hill = 0;
%                 break;
%             end
%         end
%         if hill==1
%             break;
%         end
%     end
%     ending = i;
    
    % take the noise out
    noise_portion_1 = ceil(length(raw_speech)/(starting*shift_amount));
    ft_noise_1 = abs(fft(raw_speech(1:floor(length(raw_speech)/noise_portion_1))))';
    ft_noise_1 = repmat(ft_noise_1,noise_portion_1,1);
    ft_noise_1 = ft_noise_1(:);
    ft_noise_1 = [ft_noise_1; zeros(recording_length-length(ft_noise_1),1)];
    
    noise_portion_2 = ceil(length(raw_speech)/(length(raw_speech)-ending*shift_amount));
    ft_noise_2 = abs(fft(raw_speech(ceil(length(raw_speech)*(1-1/noise_portion_2))+1:end)))';
    ft_noise_2 = repmat(ft_noise_2,noise_portion_2,1);
    ft_noise_2 = ft_noise_2(:);
    ft_noise_2 = [ft_noise_2; zeros(recording_length-length(ft_noise_2),1)];
    ft = (ft_noise_1)/2;
    
    ft2 = abs(fft(raw_speech));
    G = 1-ft./ft2;
    G = G.*(G>0);
    denoised = real(ifft(G.*fft(raw_speech)));
    
    if plotting
        figure(2)
        plot(denoised)
    end
    
    stretch_factor = (ending*shift_amount-starting*shift_amount)/length(denoised);
    stretched = zeros(length(denoised),1);
    index= 0;
    for i=1:length(stretched)
        index = index+stretch_factor;
        stretched(i) = denoised(starting*shift_amount+round(index));
    end
    
    if plotting
        figure(7)
        plot(stretched)
    end
    
    %figure(3)
    %plot(denoised)
    %audiowrite('denoised.wav', denoised, Fs);
    
%     squelch_output = [0.0001*ones(starting*shift_amount,1); raw_speech(starting*shift_amount:ending*shift_amount)];
%     squelch_output = [squelch_output; 0.0001*ones(Fs-length(squelch_output),1)];

    %figure(3)
    %plot(squelch_output)
    
    break;
    %audiowrite(strcat('preprocessed/processed_',f.name), stretched, Fs);
    
end