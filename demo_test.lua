require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
torch.setdefaulttensortype('torch.FloatTensor')

-- Functions
function normalize_global(dataset, mean, std)
  local std = std or dataset:std()
  local mean = mean or dataset:mean()
  dataset:add(-mean)
  dataset:div(std)
  return mean, std
end

function normalize_local(dataset)
  local norm_kernel = image.gaussian1D(7)
  local norm = nn.SpatialContrastiveNormalization(3,norm_kernel)
  normalized_images = norm:forward(dataset)
  dataset:copy(normalized_images)
end

IMFILE = '/media/administrator/文档/我的文件/Tencent_Model/GTSDB/resized/'
model = torch.load('./trained_models/model.t7'):cuda()
model:evaluate()

-- View kernels
--[[
local d1 = image.toDisplayTensor{input=model:get(1).weight:clone():reshape(96, 3, 11, 11),
  padding = 2,
  nrow = math.floor(math.sqrt(96)),
  symmetric = true,
}
image.display{image=d1, legend='Layer 1 filters'}
-- Save h5 file
--local myFile = hdf5.open('./trained_models/cnnkernels.h5', 'w')
--myFile:write('/DS1', d1)
--myFile:close()
--]]

local imname = '00010.ppm'
local mean_std = torch.load('MEAN_STD.t7')
local img = image.load(IMFILE..imname)
normalize_global(img, mean_std[1], mean_std[2])
normalize_local(img)

local output = model:forward(img:cuda()):float()

-- Tiling
local outputnew = torch.Tensor(1, 120, 160):zero()
for idr = 1, 15 do
	for idc = 1, 20 do
	  local tmpr = { (idr-1)*8+1, idr*8 }
	  local tmpc = { (idc-1)*8+1, idc*8 }
		outputnew[{ 1,tmpr,tmpc }] = torch.reshape(output[{ {},idr,idc }], 8, 8)
	end
end
output = outputnew

image.display(image.load(IMFILE..imname))
image.display(image.scale(output, 640, 480))

