require 'loadcaffe'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image'

proto_name = './caffe_model/model_deploy.prototxt'
model_name = './caffe_model/model.caffemodel'

print '==> Loading network'
model = loadcaffe.load(proto_name, model_name, 'cudnn')

for i = 1, 7 do
  model.modules[#model.modules] = nil -- remove several layers
end

model:add(cudnn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0, 1))
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.500000))
model:add(cudnn.SpatialConvolution(4096, 64+256, 1, 1, 1, 1, 0, 0, 1))
model:add(cudnn.Sigmoid())

-- initialization from MSR
--[[
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW * v.kH * v.nOutputPlane
      v.weight:normal(0, math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- Have to do for both backends
  init'cudnn.SpatialConvolution'
end
MSRinit(model)
--]]

-- View kernels
--[[
local d1 = image.toDisplayTensor{input=model:get(1).weight:clone():reshape(96, 3, 11, 11),
  padding = 2,
  nrow = math.floor(math.sqrt(96)),
  symmetric = true,
}
image.display{image=d1, legend='Layer 1 filters'}
--]]

-- Test model
---[[
print(model)
model = model:cuda()
local input = torch.CudaTensor(1, 3, 128, 128)
local output = model:forward(input)
print(output)
--]]

return model



