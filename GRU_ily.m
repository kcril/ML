clc;
clear;
close all;
tic
%% Loading of Data
%[AA BB CC DD EE FF GG HH JJ KK LL] =textread('G:\E-Source\014-Bilimsel Calismalar\YAYINLAR\2024\SCI_ML_kurutma\data\5_10_70.txt','%f%f%f%f%f%f%f%f%f%f%f');
[AA BB CC DD EE FF GG HH JJ KK LL] =textread('C:\Users\Gulbeyaz\Desktop\MertCan_ANN\type.txt','%f%f%f%f%f%f%f%f%f%f%f');
%[AA BB CC DD EE FF GG HH JJ KK LL] =textread('C:\Users\Gulbeyaz\Desktop\MertCan_ANN\zaman.txt','%f%f%f%f%f%f%f%f%f%f%f');
data=[AA'];

figure
plot(data)
xlabel('Zaman (s)')
ylabel('Sürtünme (N)')
title('Sürtünme kuvveti')

numTimeStepsTrain = floor(0.80*numel(data));

dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;
dataTestStandardized = (dataTest - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

XTest = dataTestStandardized(1:end-1);
YTest = dataTestStandardized(2:end);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 16;
epochs=100;
MiniBatchSize=16;
% MiniBatchSize = numTimeStepsTrain;

layers = [ ...
    sequenceInputLayer(numFeatures,'Name','sequenceInput')
%      bilstmLayer(numHiddenUnits*3,'Name','Bilstm layer','OutputMode','sequence')
%      dropoutLayer(0.2) 
%      bilstmLayer(numHiddenUnits*2,'Name','Bilstm layer','OutputMode','sequence')      
%      dropoutLayer(0.2)       
      
        gruLayer(numHiddenUnits,'Name','GRU01','OutputMode','sequence')

%       gruLayer(numHiddenUnits,'Name','GRU01','OutputMode','sequence')
    dropoutLayer(0.2)
%     gruLayer(numHiddenUnits/8,'Name','GRU02','OutputMode','sequence')
%     dropoutLayer(0.2)
%     gruLayer(numHiddenUnits/1,'Name','GRU03','OutputMode','sequence')
%     dropoutLayer(0.2)
%     gruLayer(numHiddenUnits/1,'Name','GRU04','OutputMode','sequence')
%     dropoutLayer(0.2)
    fullyConnectedLayer(numResponses,'Name','fc','WeightsInitializer','he')
    regressionLayer('Name','routput')];


options = trainingOptions('adam', ...
    'MaxEpochs',epochs, ...
    'ValidationData',{XTrain,YTrain}, ...
    'ValidationFrequency',1, ...
    'MiniBatchSize', MiniBatchSize, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

%% training
[grunet, info] = trainNetwork(XTrain,YTrain,layers,options);
% pred = classify(grunet,XTest);
% pred = predictAndUpdateState(grunet,YTrain);
grunet = predictAndUpdateState(grunet,XTrain);
[grunet,YPred] = predictAndUpdateState(grunet,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [grunet,YPred(:,i)] = predictAndUpdateState(grunet,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;
YTest = dataTest(2:end);

rmse = sqrt(mean((YPred-YTest).^2))

Accuracy = sum(YPred - YTest)/numel(YTest)*100

R_trn=corrcoef(YTest,YPred);
sure=toc;
metrikler_name={ 'MAE (birim)*'; 'MAPE(%)'; 'Ort. Hata(birim)';'MSE (birim²)'; 'RMSE (birim)';'Hesaplama süresi (s)'; 'R'; 'İterasyon adeti'};
metrikler_val={mean(abs((YPred-YTest))); 100*mean(abs(((YPred-YTest)/YTest))); mean(YPred-YTest); mean((YPred-YTest).^2); sqrt(mean((YPred-YTest).^2)); sure; R_trn(1,2); options.MaxEpochs};
METRIKLER_TRN= [metrikler_name,metrikler_val];

%% test (Open Loop Forecasting)


grunet = resetState(grunet);  % reset the network 
grunet = predictAndUpdateState(grunet,XTrain); %and initialize it by predicting on XTrain

%Şuanan kadar XTesti kullanmadık. Şimdi bir önceki döngü içerisinde YPred yerine XTesti kullanacağız. 
YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [grunet,YPred(:,i)] = predictAndUpdateState(grunet,XTest(:,i),'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;
% YTest = dataTest(2:end);
% YTrain= sig*YTrain + mu;
% XTest = sig*XTest + mu;
% XTrain= sig*XTrain + mu;

rmse = sqrt(mean((YPred-YTest).^2))
Accuracy = sum(YPred - YTest)/numel(YTest)*100

sure=toc;
name={ 'TrainErrors'};
val=[info.TrainingRMSE]'; % Test, Train
hata_yakinsama=[name;num2cell(val)];

R_test=corrcoef(YTest,YPred);

metrikler_name={ ' MAE (birim)*'; 'MAPE(%)'; 'Ort. Hata (birim)';'MSE (birim²)'; 'RMSE (birim)';'Hesaplama süresi (s)'; 'R'; 'İterasyon adeti'};
metrikler_val={mean(abs((YPred-YTest))); 100*mean(abs(((YPred-YTest)/YTest))); mean(YPred-YTest); mean((YPred-YTest).^2); sqrt(mean((YPred-YTest).^2)); sure; R_test(1,2); options.MaxEpochs};
METRIKLER_PRED= [metrikler_name,metrikler_val];

name={ 'Özellik' 'Eğitim' 'Tahmin'};
METRIKLER=[name;METRIKLER_TRN,METRIKLER_PRED(1:end,2)];



figure

plot(data(1:end-1))
% plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')

hold off
xlabel("Time Step")
ylabel("Yalpalama (rad/s)")
title("Time series response")
legend(["Observed" "Predicted"])


figure

subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Yalpalama (rad/s)")
title("Test with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Time Step")
ylabel("Error")
title("RMSE = " + rmse)

analyzeNetwork(grunet)
% confusionchart(lbl,YPred);

%% Heteroscedasticity
figure
plot(YPred,(YPred-YTest),'.')
xlabel("Fitted value (rad/s)")
ylabel("Residual (rad/s)")
%  xlabel("Fitted value (rad/s)",'Position',[mean(xlim) 0.97*max(ylim)])
%  ylabel("Residual (rad/s)",'Position',[0.99*min(xlim) 0.5*max(abs((ylim)))])
title("Heteroscedasticity")
set(gcf,'position',[200 200 800 250])
set(gcf,'color','w');

%% Closed Loop Forecasting
horizon=50;
NetOption="CNN-RNN"
nett = resetState(grunet);
    Seq = XTrain(end,:);
    [~,YPredStep] = predictAndUpdateState(nett,Seq,"MiniBatchSize",numFeatures);
    
numTimeStepsTest = size(XTest,2);

 for i = 2:numTimeStepsTest
         Seq = [YPredStep(i-1);Seq(1:end-1)];
        YPredStep(:,i) = predict(nett,Seq,"MiniBatchSize",numFeatures);
 end
    
 YPredStep = YPredStep';
YPredStep = sig*YPredStep + mu;

figure
plot(YTest,'r.-','LineWidth',1.5)
hold on
plot(YPred,'b*-','LineWidth',1.5);plot(YPredStep,'m*-','LineWidth',1.5);
legend({"Observed","Forecast Testing "+NetOption,"Forecast one step ahead and update the sequence "+NetOption},"Location","best")
ylabel("No Cases");xlabel("Month")
title("Forecasting and Testing" )
set(gca,'FontSize',12,'FontName','Adobe Kaiti Std R','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 1, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...
    'FontName', 'avantgarde', 'FontSize', 12, 'FontWeight', 'normal');

plotregression(con2seq(YTest),con2seq(YPredStep))
set(gca,'FontSize',12,'FontName','Adobe Kaiti Std R','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 1, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...
    'FontName', 'avantgarde', 'FontSize', 12, 'FontWeight', 'normal');

%beyond the horizon

    nett = resetState(nett);
    
        Seq = XTrain{end,:};
    [~,YPredStepHoriz] = predictAndUpdateState(nett,Seq,"MiniBatchSize",numFeatures);

    for i = 2:horizon
        Seq = [YPredStepHoriz(i-1);Seq(1:end-1)];
        YPredStepHoriz(:,i) = predict(nett,Seq,"MiniBatchSize",numFeatures);
    end
    
    YPredStepHoriz = YPredStepHoriz';
YPredStepHoriz = sig*YPredStepHoriz + mu;

figure
plot(YTest,'r.-','LineWidth',1.5)
hold on
plot(YPredStepHoriz,'m*-','LineWidth',1.5);
legend({"Observed","Forecast Beyond the Horizon by a "+NetOption},"Location","best")
ylabel("No Cases");xlabel("Month")
title("Predicting beyond the Horizon" )
set(gca,'FontSize',12,'FontName','Adobe Kaiti Std R','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 1, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...
    'FontName', 'avantgarde', 'FontSize', 12, 'FontWeight', 'normal');


