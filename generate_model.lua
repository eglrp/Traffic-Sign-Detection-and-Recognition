require 'loadcaffe'
require 'cutorch'
require 'cunn'
require 'cudnn'

proto_name = './caffe_model/model_deploy.prototxt'
model_name = './caffe_model/model.caffemodel'

print '==> Loading network'
model = loadcaffe.load(proto_name, model_name, 'cudnn')

for i = 1, 7 do
  model.modules[#model.modules] = nil -- remove several layers
end

-- conv7 & conv8
detect_model_all = nn.ConcatTable()
outputnum = {64+256, 1000} -- bbox, pixel, label
for i = 1, 1 do
  detect_model = nn.Sequential()
  detect_model:add(cudnn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0, 1))
  detect_model:add(cudnn.ReLU(true))
  detect_model:add(nn.Dropout(0.500000))
  detect_model:add(cudnn.SpatialConvolution(4096, outputnum[i], 1, 1, 1, 1, 0, 0, 1))
  detect_model:add(nn.Reshape(5, 120, 160))
  detect_model_all:add(detect_model)
end

-- The whole model
model:add(detect_model_all)

-- Test model
--[[
print(model)
model = model:cuda()
criterion = nn.AbsCriterion():cuda()--nn.MSECriterion():cuda()
local input = torch.CudaTensor(1, 3, 480, 640)
local output = model:forward(input)
output = output[1]
print(#output)
local f = criterion:forward(output, output)
local df_do = criterion:backward(output, output)
model:backward(input, df_do)
--]]
return model



