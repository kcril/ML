clc;clear;close all

NetOption = "CNN-RNN";    % choose the network architecture
Lag = 1: 5;% How many months to look back
ratio = 0.8;% portion to split the data in training and testing
horizon = 500; %horizon forecasting to be use in the beyong horizon section
learningrate = 0.01501;%Learning rate
solver = "adam";% solver option
MaxEpochs = 100;
mydevice = 'cpu';
 
%%Load the Data
tic;
%[AA BB CC DD EE FF GG HH JJ KK LL] =textread('G:\E-Source\014-Bilimsel Calismalar\YAYINLAR\2024\SCI_ML_kurutma\data\5_10_70.txt','%f%f%f%f%f%f%f%f%f%f%f');
[AA BB CC DD EE FF GG HH JJ KK LL] =textread('C:\Users\Gulbeyaz\Desktop\MertCan_ANN\type.txt','%f%f%f%f%f%f%f%f%f%f%f');
data=[AA'];

MiniBatchSize =floor(ratio*numel(data))

% data visualization
figure
plot(data)
xlabel("Month")
ylabel("No Cases")
title("Full Sequence of Monthly Cases of Chickenpox")
set(gca,'FontSize',12,'FontName','Adobe Kaiti Std R','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 1, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...
    'FontName', 'avantgarde', 'FontSize', 12, 'FontWeight', 'normal');

%%Split the sequence in training and testing

numStepsTraining = round(ratio*numel(data));
indexTrain = 1:numStepsTraining;
dataTrain = data(indexTrain );
indexTest = numStepsTraining+1:size(data,2);
dataTest = data(indexTest);

%% Data standardization
mu = mean(dataTrain);
sig = std(dataTrain);
TrainStandardizeddata = (dataTrain - mu) / sig;
TestStandardizeddata = (dataTest - mu) / sig;
%% Prepare Independent and Dependent Variables

% use cell array to get ready the data sequence -> Last
% Training data
XTrain = lagmatrix(TrainStandardizeddata,Lag);
XTrain = XTrain(max(Lag)+1:end,:)';
YTrain = TrainStandardizeddata(max(Lag)+1:end);
XrTrain = cell(size(XTrain,2),1);% independent variable lagged Lag months
YrTrain = zeros(size(YTrain,2),1);
for i=1:size(XTrain,2)
    XrTrain{i,1} = XTrain(:,i);
    YrTrain(i,1) = YTrain(:,i);
end

% Testing data
XTest = lagmatrix(TestStandardizeddata,Lag);
XTest = XTest(max(Lag)+1:end,:)';
YTest = TestStandardizeddata(max(Lag)+1:end);
XrTest = cell(size(XTest,2),1);% independent variable lagged Lag months
YrTest = zeros(size(YTest,2),1);
for i=1:size(XTest,2)
    XrTest{i,1} = XTest(:,i);
    YrTest(i,1) = YTest(:,i);
end

%% Create Hybrid CNN-RNN Network Architecture
numFeatures = size(XTrain,1);% it depends on the roll back window (No of features, one output)
numResponses = 1;FiltZise = 5;

 % you can follow this template and create your own Architecture
    layers = [...
        % Here input the sequence. No need to be modified
        sequenceInputLayer([numFeatures 1 1],'Name','input')
        sequenceFoldingLayer('Name','fold')

        % from here do your engeneering design of your CNN feature
        % extraction
        convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv','DilationFactor',1);
        batchNormalizationLayer('Name','bn')
        eluLayer('Name','elu')
        convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv1','DilationFactor',2);
        eluLayer('Name','elu1')
        convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv2','DilationFactor',4);
        eluLayer('Name','elu2')
        convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv3','DilationFactor',8);
        eluLayer('Name','elu3')
        convolution2dLayer(FiltZise,32,'Padding','same','WeightsInitializer','he','Name','conv4','DilationFactor',16);
        eluLayer('Name','elu4')
        averagePooling2dLayer(1,'Stride',FiltZise,'Name','pool1')

        % here you finish your CNN design and next step is to unfold and
        % flatten. Keep this part like this
        sequenceUnfoldingLayer('Name','unfold')
        flattenLayer('Name','flatten')

        % from here the RNN design. Feel free to add or remove layers
%       gruLayer(16,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(64,'Name','bilstmLayer1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.2,'Name','drop2')
        % this last part you must change the outputmode to last
        bilstmLayer(16,'OutputMode',"last",'Name','bilstmLayer2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.2,'Name','drop3')
        % here finish the RNN design

        % use a fully connected layer with one neuron because you will predict one step ahead
        fullyConnectedLayer(numResponses,'Name','fc')
        regressionLayer('Name','output')    ];

    layers = layerGraph(layers);
    layers= connectLayers(layers,'fold/miniBatchSize','unfold/miniBatchSize');

    %% Training Options
options = trainingOptions(solver, ...
        'MaxEpochs',MaxEpochs, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',learningrate, ...
        'LearnRateSchedule',"piecewise", ...
        'LearnRateDropPeriod',96, ...
        'LearnRateDropFactor',0.25, ...
        'MiniBatchSize',MiniBatchSize,...
        'Verbose',false, ...
        'Shuffle',"every-epoch",...
        'ExecutionEnvironment',mydevice,...
        'Plots','training-progress');
%% Train Hybrid Network

 [net,info] = trainNetwork(XrTrain,YrTrain,layers,options);
 
 %% Forecasting the Testing Data
 
    head(table(XrTest{1,:}))
    NextMonthPrediction = predict(net,XrTest{1,:},"ExecutionEnvironment",mydevice)
    YPred = predict(net,XrTest, ...
        "ExecutionEnvironment",mydevice,"MiniBatchSize",numFeatures);
    YPred = YPred';
    
    YPred = sig.*YPred + mu;
    YTest = sig.*YTest + mu;

[rho,pval] = corr(YPred(:),YTest(:));
% sMAPE calculation
sMAPE = 0.5*mean(abs(YPred(:)-YTest(:))./(abs(YPred(:))+abs(YTest(:))));


figure
plot(data);
hold on
idx = indexTest(1)+max(Lag):indexTest(end);
plot(idx, YPred,'.-');
xlabel("Month")
ylabel("No of Cases")
title("Forecast")
legend(["Observed" "Forecast " + NetOption],'Location',"best")
title("Forecasting vs Observed Sequence with sMAPE = " + sMAPE)
set(gca,'FontSize',12,'FontName','Adobe Kaiti Std R','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 1.1, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...
    'FontName', 'avantgarde', 'FontSize', 12, 'FontWeight', 'normal');




figure
plot(YTest,'r.-','LineWidth',1.5)
hold on
plot(YPred,'b*-','LineWidth',1.5);
legend(["Observed" "Forecast "+NetOption],"Location","best")
ylabel("No Cases");xlabel("Month")
title("Correlation: Test Sequence & Prediction = " + rho)
set(gca,'FontSize',12,'FontName','Adobe Kaiti Std R','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 1, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...
    'FontName', 'avantgarde', 'FontSize', 12, 'FontWeight', 'normal');





plotregression(con2seq(YTest),con2seq(YPred))
set(gca,'FontSize',12,'FontName','Adobe Kaiti Std R','FontWeight','bold');
set(gca, 'Box', 'on', 'LineWidth', 1, 'Layer', 'top',...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'TickDir', 'in', 'TickLength', [.015 .015],...
    'FontName', 'avantgarde', 'FontSize', 12, 'FontWeight', 'normal');


%% Forecasting Unobserved Values: the Horizon
    nett = resetState(net);
        Seq = XrTrain{end,:};
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

%% Forecasting Unobserved Values beyond the Horizon

    nett = resetState(nett);
    
        Seq = XrTrain{end,:};
    [~,YPredStepHoriz] = predictAndUpdateState(nett,Seq,"MiniBatchSize",numFeatures);

    for i = 2:horizon
        Seq = [YPredStepHoriz(i-1);Seq(1:end-1)];
        YPredStepHoriz(:,i) = predict(nett,Seq,"MiniBatchSize",numFeatures);
    end
    
    
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

sure=toc;

%% Metrikler
rmse = sqrt(mean((YPred-YTest).^2))

R_test=corrcoef(YTest,YPred);

metrikler_name={ ' MAE (birim)*'; 'MAPE(%)'; 'Ort. Hata (birim)';'MSE (birim²)'; 'RMSE (birim)';'Hesaplama süresi (s)'; 'R'; 'İterasyon adeti'};
metrikler_val={mean(abs((YPred-YTest))); 100*mean(abs(((YPred-YTest)/YTest))); mean(YPred-YTest); mean((YPred-YTest).^2); sqrt(mean((YPred-YTest).^2)); sure; R_test(1,2); options.MaxEpochs};
METRIKLER_PRED= [metrikler_name,metrikler_val];


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

analyzeNetwork(net)