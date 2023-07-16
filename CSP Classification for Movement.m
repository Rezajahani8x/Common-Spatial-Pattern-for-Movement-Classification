            %% Section 1
    %% Pre steps
load('hw3-1.mat');
TrainData_class1 = zero_mean(TrainData_class1);
TrainData_class2 = zero_mean(TrainData_class2);
TestData = zero_mean(TestData);
[M,T,N] = size(TrainData_class1);
% T = 256;
% M = 30;
% N = 60;

    %% Part 1 - Alef)
Rx1 = AutoCorr(TrainData_class1);
Rx2 = AutoCorr(TrainData_class2);
[Wcsp,Dcsp] = eig(inv(Rx2)*Rx1);
Dcsp_vals = diag(Dcsp);
Dcsp_vals = sort(Dcsp_vals,'descend');
Wcsp_temp = Wcsp;
for i=1:M
    [~,col] = find(Dcsp==Dcsp_vals(i)); 
    Wcsp(:,i) = Wcsp_temp(:,col);
end
Wcsp = normc(Wcsp);
[Var_class1,X_train_class1] = CSP(Wcsp,TrainData_class1);
[Var_class2,X_train_class2] = CSP(Wcsp,TrainData_class2);

    %% Part 3 - jim)
main_filter_index = [1:7 24:30];
Xc1 = Var_class1(main_filter_index,:);
Xc2 = Var_class2(main_filter_index,:);
m1 = 1/N * sum(Xc1,2);
m2 = 1/N * sum(Xc2,2);
C1 = Cov(Xc1,m1); 
C2 = Cov(Xc2,m2);
Mat1 = (m1-m2)*transpose(m1-m2);
Mat2 = C1 + C2;
[Q,D] = eig(inv(Mat2)*Mat1);
[~,idx] = max(max(D));
W_LDA = normc(Q(:,idx));
miu_1 = transpose(W_LDA)*m1;
miu_2 = transpose(W_LDA)*m2;
c = 1/2 * (miu_1 + miu_2);
% disp('W_LDA=');
% disp(W_LDA);
% disp('c=');
% disp(c);

    %% Part 4 - dal)
W = Wcsp(:,main_filter_index);
y = CSP_Model(W,W_LDA,c,TestData);

    %% Part 5 - he)
N_test = length(TestLabel);
correct_estimations = (y == TestLabel);
num_correct_estimations = sum(correct_estimations);
accuracy = num_correct_estimations/N_test;
test_num = 1:N_test;
figure(3);
scatter(test_num,TestLabel,'b'); hold on; grid on;
scatter(test_num,y,'r');
xlim([1,N_test]);
ylim([0,3]);
xlabel("Test Data Number");
ylabel("True & Estimated class");
title("True & Estimated Labels per Data Number");
legend('True Labels','Estimated Labels');
disp("Accuracy of the Model:");
disp(accuracy);

    %% Local Necessary Functions 
function y = zero_mean(x)
    size_x = size(x);
    y = zeros(size_x);
    N = size_x(3);
    for i=1:N
        y(:,:,i) = x(:,:,i) - mean(x(:,:,i),2);
    end
end

function Rx = AutoCorr(X)
    size_X = size(X);
    N = size_X(3);
    M = size_X(1);
    Rx = zeros(M,M);
    for n=1:N
       x = X(:,:,n);
       Rx = Rx + x*transpose(x);
    end
    Rx = 1/N * Rx;
end

function [Var_Mat,Y] = CSP(Wcsp,X)
    [~,T,N] = size(X);
    [~,L] = size(Wcsp);
    Y = zeros(L,T,N);
    Var_Mat = zeros(L,N);
    for n=1:N
       x = X(:,:,n);
       Y(:,:,n) = transpose(Wcsp) * x;
       Var_Mat(:,n) = var(transpose(Y(:,:,n)));
    end
end

function C = Cov(Xc,m)
    [M,N] = size(Xc);
    C = zeros(M,M);
    for i=1:N
        C = C + (Xc(:,i)-m)*transpose((Xc(:,i)-m));
    end
    C = 1/N * C;
end

function Y = CSP_Model(Wcsp,Wlda,c,X_test)
    X = X_test;
    [x,~] = CSP(Wcsp,X);
    y = transpose(Wlda) * x;
    Y(y>=c) = 2;
    Y(y<c) = 1;
end