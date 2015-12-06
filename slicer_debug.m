% slicer_debug.m
% CS229 Project
% Yuki Inoue
%
% Usage:
% Same structure as the slicer.m, except it doesn't exhaustively look
% through the files in /recording.
% Instead, the user must specify the filename (f_name) of the wav file
% he/she wishes to analyze.
% slice output can be enabled by setting the "outputting" flag appropriately.
% Should be used to "debug" those audio files that were not successfully
% sliced by the slicer.m. Use the plots to help you.

% The first plot shows the raw audio file, and the dotted lines are where
% the script estimates that a word is being spoken.
% The second plot shows the power of the audio signal (a section of raw signal squared)
% Essencially, this plot should have high values whenever the volume of the
% input signal is higher.


clear all;
outputting = 0;
shift_per_sec = 32;
l=3;
nse = [26, 10, 14];
symbol_list = cellstr(['exclamation'; 'at         '; 'hash       '; 'dollar     ';
               'percent    '; 'caret      '; 'and        '; 'star       '; 
               'underscore '; 'dash       '; 'comma      '; 'period     '; 
               'question   '; 'tilde      ']);
num_samples_expected = nse(l);

f_name = 'special14.wav';
[sound_sample, Fs] = wavread(strcat('recording/',f_name));
%sound(sound_sample,Fs)
%sound_sample = sound_sample(1:7*10^5);
figure(1)
plot(sound_sample)

n = round(length(sound_sample)/Fs*shift_per_sec);
sec_sub = round(Fs/4);
shift_amount = round(length(sound_sample)/n);
sound_sample_sq = sound_sample.^2;
section_power = zeros(n,1);
for i=1:n
    start_index = (i-1)*shift_amount+1;
    section_power(i) = sum(sound_sample_sq(start_index:min(start_index+sec_sub,length(sound_sample_sq))));
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
end
%div_center

for i=1:length(div_center)
    d = div_center(i);
    hx = graph2d.constantline(d, 'LineStyle',':', 'Color',[0 0 0]);
    changedependvar(hx,'x');
end

div_center = round(div_center);

figure(2)
plot(section_power)

if(outputting)
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
        wavwrite(sample,Fs,strcat('slicedup/',val,'_',f_name));
    end
end