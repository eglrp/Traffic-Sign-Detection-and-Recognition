require 'xlua'
require 'optim'
require 'pl'
require 'cudnn'
require 'gnuplot'
local c = require 'trepl.colorize'


opt = lapp[[
   -s,--save										(default "logs")                       subdirectory to save logs
   -b,--batchSize								(default 1)                            batch size
   -r,--learningRate            (default 1e-4)                         learning rate
   --learningRateDecay          (default 1e-7)                         learning rate decay
   --weightDecay                (default 5e-4)                         weightDecay
   -m,--momentum                (default 0.9)                          momentum
   --epoch_step                 (default 25)                           epoch step
   --max_epoch                  (default 400)                          maximum number of iterations
   --type                       (default cuda)                         cuda or double
]]

print(opt)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

print(c.red'==>'..c.red' load model')
model = dofile('generate_model.lua'):cuda()
parameters, gradParameters = model:getParameters()
print(model)

print(c.red'==>' ..c.red' setting criterion')
criterion = nn.AbsCriterion():cuda()--nn.MSECriterion():cuda()

print(c.red'==>'..c.red' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

print(c.red'==>'..c.red' load labels')
dofile('demo_createlabel.lua')

function train()

  model:training()

  local cost = {}
  local inputs = torch.Tensor(opt.batchSize, 3, 480, 640):zero()
  local targets = torch.Tensor(opt.batchSize, 5, 120, 160):zero()
  if opt.type == 'double' then inputs = inputs:double()
  elseif opt.type == 'cuda' then inputs = inputs:cuda() targets = targets:cuda() end

  --epoch = epoch or 1
  -- drop learning rate every "epoch_step" epochs
  --if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  -- shuffle at each epoch
  local trainnum = #GTRUTH
  shuffle = torch.randperm(trainnum)

  for t = 1, trainnum, opt.batchSize do
    -- Disp progress
    xlua.progress(t, trainnum)

    -- Create mini batch
    local indx = 1
    for i = t, math.min(t+opt.batchSize-1, trainnum) do
      -- Load new sample
      local tmp = GTRUTH[shuffle[i]]
      local imname = tmp[1]
      local indCol = tmp[2]
      local indRow = tmp[3]
      local disCol = tmp[4]
      local disRow = tmp[5]
      local W = tmp[6]
      local H = tmp[7]
      local img = image.load(IMFILE..imname)
      inputs[{ indx,{},{},{} }] = image.scale(img, 640, 480)
      targets[{ indx, 1, indRow, indCol}] = 1
      targets[{ indx, 2, indRow, indCol}] = disCol
      targets[{ indx, 3, indRow, indCol}] = disRow
      targets[{ indx, 4, indRow, indCol}] = W
      targets[{ indx, 5, indRow, indCol}] = H
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
      local output = model:forward(inputs)
      output = output[1]
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
    --trainlcc, trainsrocc, testlcc, testsrocc, testtarg, testout = test()
  end
end

--[[
torch.save('./trained_models/NRIQA_MODEL_1024to1.t7'..DISTYPE, model )
torch.save('./trained_models/LCC_TRAIN_1024to1.t7'..DISTYPE, LCC_train)
torch.save('./trained_models/LCC_TEST_1024to1.t7'..DISTYPE, LCC_test)
torch.save('./trained_models/SROCC_TRAIN_1024to1.t7'..DISTYPE, SROCC_train)
torch.save('./trained_models/SROCC_TEST_1024to1.t7'..DISTYPE, SROCC_test)
--]]

