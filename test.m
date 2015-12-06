alpha = 1;
num = 2;
special = 3;

type = alpha;

num_trials = 1;

s = [];

symbol_list = cellstr(['at         '; 'hash       '; 'dollar     ';
               'percent    '; 'caret      '; 'and        '; 'star       '; 
               'dash       '; 'comma      '; 'period     '; 
               'question   '; 'tilde      ']);

symbol_list = cellstr(['at'; '#       '; '$     ';
               '%    '; 'caret'; 'and        '; '*       '; 
               '-       '; ',      '; '.     '; 
               '?   '; 'tilde     ']);
               
if type==alpha
    labeling_size = 26;
    starting_char = 'a';
elseif type==num
    labeling_size = 10;
    starting_char = '0';
else
    labeling_size = 12 %14;
    starting_char = '0';
end
s = zeros(labeling_size,1);
for i=1:num_trials
    if type==alpha
        systemCommand = ['python test_2.py'];
    elseif type==num
        systemCommand = ['python test_2_num.py'];
    else
        systemCommand = ['python test_2_special.py'];
    end
    [status, result] = system(systemCommand);

    array_start = find(result=='[');
    array_start = array_start(end)+1;
    array_end = find(result==']')-1;
    array_end = array_end(end)-1;
    num_str = result(array_start:array_end);

    num_str=strrep(num_str,',','');
    s_temp=textscan(num_str,'%f');
    s = s+s_temp{1};
end

if type==alpha || type==num
  label = 0:labeling_size-1;
  label = (label + starting_char)';
else
  label = symbol_list
end

figure
bar(s)
title('Labeling Failure')
str = cellstr(char(label));
set(gca, 'XTickLabel',str, 'XTick',1:numel(str))
axis([0 labeling_size+1 0 1])

sum(s)/length(s)
