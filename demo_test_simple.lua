require 'image'
require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'gnuplot'
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

local imname = '00123.ppm'
local mean_std = torch.load('MEAN_STD.t7')
--local img_old = image.load(IMFILE..imname)

local img_old = image.load('./testimg1.jpg')
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
local th = 0.6
local output = output:sum(1):squeeze()
output = output / output:max()
output = output:ge(th)
--image.display(output)

-- Merge and generate bbox
-- bwlabel
local location_r = output:nonzero()
local location_c = output:t():nonzero()
local len = location_r:size(1)
local diff_r = location_r[{ {},1 }]
local diff_c = location_c[{ {},1 }]
diff_r = location_r[{ {2,len},{} }] - location_r[{ {1,len-1},{} }]
diff_c = location_c[{ {2,len},{} }] - location_c[{ {1,len-1},{} }]
local bwlabel = torch.Tensor(2, output:size(1), output:size(2)):zero()
local indr, indc = 1, 1
for i = 2, len do
	-- Consider row
	if (diff_r[i-1][1]==0 or diff_r[i-1][1]==1) then
		bwlabel[{1,location_r[i][1],location_r[i][2]}] = indr
	else
		indr = indr + 1
		bwlabel[{1,location_r[i][1],location_r[i][2]}] = indr
	end
	-- Consider col
	if (diff_c[i-1][1]==0 or diff_c[i-1][1]==1) then
		bwlabel[{2,location_c[i][2],location_c[i][1]}] = indc
	else
		indc = indc + 1
		bwlabel[{2,location_c[i][2],location_c[i][1]}] = indc
	end
end
bwlabel = torch.cmul(bwlabel[1],bwlabel[1]) + bwlabel[2] - 1
bwlabel[bwlabel:eq(-1)] = 0
local object_num = bwlabel:max()
gnuplot.imagesc(bwlabel)

-- Plot bbox
local tmp = img_old[2]
tmp[output] = 1
img_old[2] = tmp
for i = 1, object_num do
	local tmp = bwlabel:eq(i):nonzero()
	if (tmp:dim() > 0) then
		local top = tmp[{ {},1 }]:min()
		local bottom = tmp[{ {},1 }]:max()
		local left = tmp[{ {},2 }]:min()
		local right = tmp[{ {},2 }]:max()
		draw.drawBox(img_old, top, bottom, left, right, 2, {math.random(), math.random(), math.random()})
		d1 = image.display{image = img_old, win = d1}
	end
end


















