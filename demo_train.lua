require 'xlua'
require 'optim'
require 'pl'
require 'cudnn'
require 'gnuplot'
local c = require 'trepl.colorize'


opt = lapp[[
   -s,--save                             (default "logs")                                    subdirectory to save logs
   -b,--batchSize                    (default 2)                                          batch size
   -r,--learningRate               (default 1e-4)                                      learning rate
   --learningRateDecay        (default 1e-7)                                      learning rate decay
   --weightDecay                   (default 5e-4)                                      weightDecay
   -m,--momentum               (default 0.9)                                         momentum
   --epoch_step                      (default 25)                                         epoch step
   --model                              (default nin_imagenet_pretrain)      model name
   --max_epoch                      (default 300)                                      maximum number of iterations
   --type                                  (default cuda)                                     cuda or double
]]

print(opt)

print(c.blue '==>' ..' configuring model')
dofile('data_preprocess_changeform.lua')
model = dofile('models/'..opt.model..'.lua'):cuda()
--model = torch.load('./trained_models/NRIQA_MODEL_NewForm.t7'):cuda()

print(model)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters, gradParameters = model:getParameters()

print(c.blue'==>' ..' setting criterion')
criterion = nn.MSECriterion():cuda()

print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()

  local cost = {}
  local inputs = torch.Tensor(opt.batchSize, 3, 224, 224)
  local targets = torch.Tensor(opt.batchSize)
  if opt.type == 'double' then inputs = inputs:double()
  elseif opt.type == 'cuda' then inputs = inputs:cuda() targets = targets:cuda() end

  model:training()
  --epoch = epoch or 1
  -- drop learning rate every "epoch_step" epochs
  --if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  -- shuffle at each epoch
  shuffle = torch.randperm(#trainNameList)

  for t = 1, #trainNameList, opt.batchSize do
    -- disp progress
    xlua.progress(t, #trainNameList)

    -- create mini batch
    local indx = 1
    for i = t, math.min(t+opt.batchSize-1, #trainNameList) do
      -- load new sample
      local tpname = trainNameList[shuffle[i]]
      tpname = './datas/LIVE_imagepatches/'..tpname
      inputs[{ indx, {},{},{} }] = preprocess(image.load(tpname), img_mean)
      targets[indx] = trainLabels[shuffle[i]]
      indx = indx + 1
    end

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end
      -- reset gradients
      gradParameters:zero()
      -- evaluate function for complete mini batch
      -- estimate f
      local output = model:forward(inputs):squeeze()
      local f = criterion:forward(output, targets)
      table.insert(cost, f)
      -- estimate df/dW
      local df_do = criterion:backward(output, targets)
      model:backward(inputs, df_do)
      -- normalize gradients and f(X)
      gradParameters:div(inputs:size(1))
      -- return f and df/dX
      return f, gradParameters
    end

    optim.sgd(feval, parameters, optimState)

    gnuplot.plot(torch.Tensor(cost))

  end
end


function test()

  model:evaluate()

  -- Test on train, test and validate images
  local function testfunc_train(NameList, Labels)
    local batchsize = 20
    local sizeall = batchsize - math.fmod(#NameList, batchsize) + #NameList
    local inputs = torch.Tensor(batchsize, 3, 224, 224)
    local targets = torch.Tensor(sizeall)
    local outputs = torch.Tensor(sizeall)
    local indxall = 1;
    if opt.type == 'double' then inputs = inputs:double()
    elseif opt.type == 'cuda' then inputs = inputs:cuda()  end

    for t = 1, sizeall, batchsize do
      xlua.progress(t, sizeall)
      local indx = 1
      for i = t, math.min(t+batchsize-1, sizeall) do
        -- load new sample
        local tpname = NameList[i]
        if (tpname ~= nil) then
          tpname = './datas/LIVE_imagepatches/'..tpname
          inputs[{ indx, {},{},{} }]= preprocess(image.load(tpname), img_mean)
          targets[indxall] = Labels[i]
        else
          inputs[{ indx,{},{},{} }]= inputs[{ 1,{},{},{} }]
          targets[indxall] = Labels[1]
        end
        indx = indx + 1
        indxall  = indxall + 1
      end
      outputs[{ {t,t+batchsize-1} }] = model:forward(inputs):double():squeeze()
    end
    targets = targets[{ {1,#NameList} }]
    outputs = outputs[{ {1,#NameList} }]
    local lcc = lcc(targets, outputs)
    local srocc = srocc(targets, outputs)
    return lcc, srocc, targets, outputs
  end

  local function testfunc_test(NameList, Labels)
    local batchsize = 1
    local sizeall = batchsize - math.fmod(#NameList, batchsize) + #NameList
    local inputs = torch.Tensor(batchsize, 3, 448, 448)
    local targets = torch.Tensor(sizeall)
    local outputs = torch.Tensor(sizeall)
    local indxall = 1;
    if opt.type == 'double' then inputs = inputs:double()
    elseif opt.type == 'cuda' then inputs = inputs:cuda()  end

    for t = 1, sizeall, batchsize do
      xlua.progress(t, sizeall)
      local indx = 1
      for i = t, math.min(t+batchsize-1, sizeall) do
        -- load new sample
        local tpname = NameList[i]
        if (tpname ~= nil) then
          tpname = './datas/LIVE_images/'..tpname
          inputs[{ indx, {},{},{} }]= preprocess_large(image.load(tpname), img_mean_large)
          targets[indxall] = Labels[i]
        else
          inputs[{ indx,{},{},{} }]= inputs[{ 1,{},{},{} }]
          targets[indxall] = Labels[1]
        end
        indx = indx + 1
        indxall  = indxall + 1
      end
      outputs[{ {t,t+batchsize-1} }] = model:forward(inputs):double():squeeze():mean()
    end
    targets = targets[{ {1,#NameList} }]
    outputs = outputs[{ {1,#NameList} }]
    local lcc = lcc(targets, outputs)
    local srocc = srocc(targets, outputs)
    return lcc, srocc, targets, outputs
  end

  local trainlcc, trainsrocc = testfunc_train(trainNameList, trainLabels)
  local testlcc, testsrocc, targets, outputs = testfunc_test(testNameList, testLabels)

  print('LCC train: '..trainlcc..' SROCC train: '..trainsrocc)
  print('LCC test: '..testlcc..' SROCC test: '..testsrocc)

  return trainlcc, trainsrocc, testlcc, testsrocc, targets, outputs
end


--local LCC_train = {}
--local SROCC_train = {}
--local LCC_test = {}
--local SROCC_test = {}

for i = 1,  opt.max_epoch do
  train()
  if (i  == 1 or math.fmod(i, 10) == 0) then
    print('Epoch '..i)
    trainlcc, trainsrocc, testlcc, testsrocc, testtarg, testout = test()
    --    table.insert(LCC_train, trainlcc)
    --    table.insert(SROCC_train, trainsrocc)
    --    table.insert(LCC_test, testlcc)
    --    table.insert(SROCC_test, testsrocc)

    --    local tmptesttarg = torch.Tensor.resize(testtarg, (#testtarg)[1]/9, 9):mean(2):squeeze()
    --    local tmptestout = torch.Tensor.resize(testout, (#testout)[1]/9, 9):mean(2):squeeze()
    --    local lcc_avg = lcc(tmptesttarg, tmptestout)
    --    local srocc_avg = srocc(tmptesttarg, tmptestout)
    --    print('LCC test avg mean: '..lcc_avg..', SROCC test avg mean: '..srocc_avg)
    --    tmptesttarg = testtarg:median(2):squeeze()
    --    tmptestout = testtarg:median(2):squeeze()
    --    lcc_avg = lcc(tmptesttarg, tmptestout)
    --    srocc_avg = srocc(tmptesttarg, tmptestout)
    --    print('LCC test avg median: '..lcc_avg..', SROCC test avg median: '..srocc_avg)
  end
end

torch.save('./trained_models/NRIQA_MODEL_1024to1.t7'..DISTYPE, model )
torch.save('./trained_models/LCC_TRAIN_1024to1.t7'..DISTYPE, LCC_train)
torch.save('./trained_models/LCC_TEST_1024to1.t7'..DISTYPE, LCC_test)
torch.save('./trained_models/SROCC_TRAIN_1024to1.t7'..DISTYPE, SROCC_train)
torch.save('./trained_models/SROCC_TEST_1024to1.t7'..DISTYPE, SROCC_test)


