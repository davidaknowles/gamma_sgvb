function [e] = log_sum_exp(v)

c=max(v);
% max_ex= max(v);
% max_ex_abs=max(abs(v));

% if (max_ex_abs>max_ex)
%     c = min(v);
% else
%     c=max_ex;
% end


e = log(sum(exp(v-c)))+c;

end