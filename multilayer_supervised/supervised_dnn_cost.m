function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
for i=1:numHidden+1
    if(i==1)
        hAct{1}=bsxfun(@plus,stack{1}.W*data,stack{1}.b);
    else
        hAct{i}=bsxfun(@plus,stack{i}.W*hAct{i-1},stack{i}.b);
    end
    if(i<numHidden+1)
        switch ei.activation_fun
            case 'logistic'
                hAct{i}=sigmoid(hAct{i});
            case 'relu'
                hAct{i}=relu(hAct{i});
            case 'tanh'
                hAct{i}=tanh(hAct{i});
        end
    end
end
% hAct{1}=sigmoid(bsxfun(@plus,stack{1}.W*data,stack{1}.b));
% for i=2:numel(hAct)
%     if(i<numel(hAct))
%         hAct{i}=sigmoid(bsxfun(@plus,stack{i}.W*hAct{i-1},stack{i}.b));
%     else
%         hAct{i}=bsxfun(@plus,stack{i}.W*hAct{i-1},stack{i}.b);
%     end
% end
pred_prob=bsxfun(@rdivide,exp(hAct{end}),sum(exp(hAct{end}),1));

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end

%% compute cost
%%% YOUR CODE HERE %%%
ind=sub2ind(size(pred_prob),labels',1:size(pred_prob,2));
ceCost=-sum(log(pred_prob(ind)));

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
labels_full=full(sparse(labels,1:numel(labels),1));
delta=cell(numHidden+1,1);
delta{numHidden+1}=-(labels_full-pred_prob);
for i=numHidden:-1:1
    switch ei.activation_fun
        case 'logistic'
            delta{i}=stack{i+1}.W'*delta{i+1}.*(hAct{i}.*(1-hAct{i}));
        case 'relu'
            delta{i}=stack{i+1}.W'*delta{i+1}.*(hAct{i}>0);
        case 'tanh'
            delta{i}=stack{i+1}.W'*delta{i+1}.*(1-hAct{i}.^2);
    end
end
for i=1:numHidden+1
    if(i==1)
        gradStack{i}.W=delta{i}*data';
    else
        gradStack{i}.W=delta{i}*hAct{i-1}';
    end
    gradStack{i}.b=sum(delta{i},2);
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost=0;
for i=1:numel(stack)
    wCost=wCost+1/2*ei.lambda*sum(stack{i}.W(:).^2);
end
cost=ceCost+wCost;
for i=1:numHidden
    gradStack{i}.W=gradStack{i}.W+ei.lambda*stack{i}.W;
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end