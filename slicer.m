% slicer.m
% CS229 Project
% Yuki Inoue
%
% Usage:
% Place this script in a directory with folders named /recording and /slicedup
% Name the wav files with prefix alpha, num, or special, depending on what
% kind of file they are, and place them inside of the folder named /recording
% The script will look into the folder named /recording, slices up the audio, and
% outputs the result in the folder named /slicedup

clear all;
plotting = 0;
shift_per_sec = 32;
nse = [26, 10, 14];
symbol_list = cellstr(['exclamation'; 'at         '; 'hash       '; 'dollar     ';
               'percent    '; 'caret      '; 'and        '; 'star       '; 
               'underscore '; 'dash       '; 'comma      '; 'period     '; 
               'question   '; 'tilde      ']);

['!','@','#','$','%','^','&','*','_','-',',','.','?','~'];

for l=3:3
    num_samples_expected = nse(l);

    files = [];
    if(l==1)
        files = dir('recording/alpha*.wav');
    elseif(l==2)
        files = dir('recording/num*.wav');
    else
        files = dir('recording/*special*.wav');
    end
    for f=files'
        disp(f.name);
        [sound_sample, Fs] = audioread(strcat('recording/',f.name));
        %sound(sound_sample,Fs)
        if(plotting)
            figure(1)
            plot(sound_sample)
        end

        n = round(length(sound_sample)/Fs*shift_per_sec);
        sec_sub = round(Fs/4);
        shift_amount = round(length(sound_sample)/n);
        sound_sample_sq = sound_sample.^2;
        section_power = zeros(n,1);
        for i=1:n
            start_index = (i-1)*shift_amount+1;
            section_power(i) = sum(sound_sample_sq(start_index:start_index+sec_sub));
        end

        div_center = [];
        for threshold=100:-2:2
            div_center = [];
            above = 0;
            i_index = 0;
            i = 1;
            while(i<n)
                if(above==0 && section_power(i)>threshold)
                    above = 1;
                    i_index = i;
                end
                if(above==1 && section_power(i)<threshold)
                    above = 0;
                    div_center = [div_center; round(i_index+i)/2];
                    i = round(i+shift_per_sec/4);
                    while(i<n && section_power(i)>threshold)
                        i = i+1;
                    end
                end
                i = i+1;
            end
            div_center = shift_amount*div_center+sec_sub/2;
            if(length(div_center)==num_samples_expected)
                break;
            end
        end
        if(length(div_center)~=num_samples_expected)
            disp('sample extraction failed');
            continue;
        end
        %div_center

        if(plotting)
            for i=1:length(div_center)
                d = div_center(i);
                hx = graph2d.constantline(d, 'LineStyle',':', 'Color',[0 0 0]);
                changedependvar(hx,'x');
            end
        end

        div_center = round(div_center);

        figure(2)
        plot(section_power)
        break;

        % output the samples to files
        half_sec = round(Fs/2);
        for i=1:num_samples_expected
            sample = sound_sample(max(1,div_center(i)-half_sec):min(length(sound_sample),div_center(i)+half_sec));
            val = 0;
            if(l==1)
                val = char(96+i);
            elseif(l==2)
                val = char(47+i);
            else
                val = char(symbol_list(i));
            end
            %audiowrite(strcat('slicedup_2/',val,'_',f.name),sample,Fs);
        end
    end
end