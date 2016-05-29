require 'loadcaffe'
require 'cutorch'
require 'cunn'
require 'cudnn'

proto_name = 'model_deploy.prototxt'
model_name = 'model.caffemodel'

print '==> Loading network'
model = loadcaffe.load(proto_name, model_name, 'cudnn')

print(model)

for i = 1, 7 do
  model.modules[#model.modules] = nil -- remove several layers
end

-- conv7 & conv8
detect_model_all = nn.ConcatTable()
outputnum = torch.Tensor({256, 128, 1000}) -- bbox, pixel, label
for i = 1, 3 do
  detect_model = nn.Sequential()
  detect_model:add(cudnn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0, 1))
  detect_model:add(cudnn.ReLU(true))
  detect_model:add(nn.Dropout(0.500000))
  detect_model:add(cudnn.SpatialConvolution(4096, outputnum[i], 1, 1, 1, 1, 0, 0, 1))
  detect_model_all:add(detect_model)
end

-- the whole model
model:add(detect_model_all)

print(model)
print(model:cuda():forward(torch.CudaTensor(1, 3, 355, 355)))

return model



