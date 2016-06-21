require 'image'
require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'gnuplot'
require 'imgraph'
draw = require 'draw'
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

IMFILE = '/media/lab/52347F6D347F5349/detection_model/resized/'
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

local imname = '00343.ppm'
local mean_std = torch.load('MEAN_STD.t7')
--local img_old = image.load(IMFILE..imname)

local img_old = image.load('./testimg4.jpg')
img_old = image.scale(img_old, 640, 480)

local img = img_old:clone()
normalize_global(img, mean_std[1], mean_std[2])
normalize_local(img)

local output = model:forward(img:cuda()):float()

-- Tiling
local outputnew = torch.Tensor(5, 120, 160):zero()
for idr = 1, 15 do
	for idc = 1, 20 do
	  local tmpr = { (idr-1)*8+1, idr*8 }
	  local tmpc = { (idc-1)*8+1, idc*8 }
		outputnew[{ {},tmpr,tmpc }] = torch.reshape(output[{ {},idr,idc }], 5, 8, 8)
	end
end
output = image.scale(outputnew, 640, 480)

-- Sum and thresh
local th = 0.4
local output = output:sum(1):squeeze() / 5
output = output:ge(th)
--image.display(output)

-- Merge and generate bbox
-- bwlabel
local graph = imgraph.graph(output:float())
graph = imgraph.connectcomponents(graph, 0.1) + 10
graph = torch.cmul(graph, output:float())
--image.display(graph)
local location = graph:nonzero()
local val = graph[location[1][1]][location[1][2]]
local indx = {}
table.insert(indx, val)
for i = 1, location:size(1) do
	local tmp = graph[location[i][1]][location[i][2]]
	local flag = true
	for j = 1, #indx do
		if (tmp == indx[j]) then
			flag = false
			break
		end
	end
	if (flag == true) then
		val = tmp
		table.insert(indx, val)
	end
end
print(#indx)

-- Plot bbox
local tmp = img_old[2]
tmp[output] = 1
img_old[2] = tmp
local bias = 2
for i = 1, #indx do
	local tmp = graph:eq(indx[i]):nonzero()
	if (tmp:dim() > 0) then
		local top = tmp[{ {},1 }]:min() - bias
		local bottom = tmp[{ {},1 }]:max() + bias
		local left = tmp[{ {},2 }]:min() - bias
		local right = tmp[{ {},2 }]:max() + bias
		draw.drawBox(img_old, top, bottom, left, right, 2, {math.random(), math.random(), math.random()})
		d1 = image.display{image = img_old, win = d1}
	end
end

















