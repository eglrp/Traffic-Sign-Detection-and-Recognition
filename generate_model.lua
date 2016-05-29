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
outputnum = {128, 256, 64000} -- bbox, pixel, label
reshapesize = {2, 4, 1000}
for i = 1, 2 do
  detect_model = nn.Sequential()
  detect_model:add(cudnn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0, 1))
  detect_model:add(cudnn.ReLU(true))
  detect_model:add(nn.Dropout(0.500000))
  detect_model:add(cudnn.SpatialConvolution(4096, outputnum[i], 1, 1, 1, 1, 0, 0, 1))
  -- Generate maks 120 * 160
  detect_model:add(nn.Reshape(reshapesize[i], 120, 160))
  detect_model_all:add(detect_model)
end

-- The whole model
model:add(detect_model_all)



print(model)
print(model:cuda():forward(torch.CudaTensor(1, 3, 480, 640)))

return model



