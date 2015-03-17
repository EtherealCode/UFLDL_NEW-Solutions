global params;
params.m=10;
params.patchWidth=2; % width of a patch
params.n=params.patchWidth^2; % dimensionality of input to RICA
params.lambda = 0.0005; % sparsity cost
params.numFeatures = 10; % number of filter banks to learn
params.epsilon = 1e-2; % epsilon to use in square-sqrt nonlinearity
data = loadMNISTImages('../common/train-images-idx3-ubyte');
patches = samplePatches(data,params.patchWidth,params.m);
patches = zca2(patches);
m = sqrt(sum(patches.^2) + (1e-8));
x = bsxfunwrap(@rdivide,patches,m);
randTheta = randn(params.numFeatures,params.n)*0.01; % 1/sqrt(params.n);
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2));
randTheta = randTheta(:);
[~,grad]=softICACost(randTheta,x,params);
numgrad=zeros(size(grad));
eps=0.001;
for i=1:length(randTheta)
    randThetaplus=randTheta;
    randThetaminus=randTheta;
    randThetaplus(i)=randThetaplus(i)+eps;
    randThetaminus(i)=randThetaminus(i)-eps;
    numgrad(i)=(softICACost(randThetaplus,x,params)-softICACost(randThetaminus,x,params))/(2*eps);
end
disp([grad numgrad]);
diff=norm(grad-numgrad)/norm(grad+numgrad);
disp(diff);