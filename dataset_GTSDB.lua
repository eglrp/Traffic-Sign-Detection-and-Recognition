require 'torch'
require 'image'
require 'gnuplot'
require 'nn'
draw = require 'draw'
torch.setdefaulttensortype('torch.FloatTensor')

GTFILE = '/media/administrator/文档/我的文件/Tencent_Model/GTSDB/FullIJCNN2013/gt.txt'
IMFILE = '/media/administrator/文档/我的文件/Tencent_Model/GTSDB/FullIJCNN2013/'
IMREFILE = '/media/administrator/文档/我的文件/Tencent_Model/GTSDB/resized/'

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
  local batch = 10 -- Can be reduced if you experience memory issues
  local dataset_size = dataset:size(1)
  for i=1,dataset_size,batch do
    local local_batch = math.min(dataset_size,i+batch) - i
    local normalized_images = norm:forward(dataset:narrow(1,i,local_batch))
    dataset:narrow(1,i,local_batch):copy(normalized_images)
  end
end
-------------------------------------------------------------------------------------

-- Create ground truth
GTRUTH = {}
local f = assert(io.open(GTFILE, 'r'))
local imnameold = '00000.ppm'
local tmpgtruth = {}

while(1) do
  local line = f:read("*line")
  if (line == nil) then
    break
  end 
  
  -- Get ground truth
  local gtruth = string.split(line, ';')
  local imname = gtruth[1]
	local leftcol = tonumber(gtruth[2])
	local toprow = tonumber(gtruth[3])
	local rightcol = tonumber(gtruth[4])
	local bottomrow = tonumber(gtruth[5])
  
  -- Get image size
  local imrow = 800
  local imcol = 1360
  local indleft = torch.round(leftcol / imcol * 640)
  local indright = torch.round(rightcol / imcol * 640)
  local indtop = torch.round(toprow / imrow * 480)
  local indbottom = torch.round(bottomrow / imrow * 480)
  
  if (imname == imnameold) then
		table.insert(tmpgtruth, {indleft, indright, indtop, indbottom})
  else
    table.insert(GTRUTH, {imnameold, tmpgtruth})
    tmpgtruth = {}
  	imnameold = imname
  	table.insert(tmpgtruth, {indleft, indright, indtop, indbottom})
  end
end

print(GTRUTH)

-- Create image patches
local patchsize = 128
local step = 64
local masksize = patchsize / 4
local totalnum = 3366
IMAGEPATCH = torch.Tensor(totalnum, 3, patchsize, patchsize)
IMAGEMASK = torch.Tensor(totalnum, masksize, masksize)
PATCHSCORE = torch.Tensor(totalnum, 1)
local indpatch = 1

for i = 1, #GTRUTH do

	local tmpgruth = GTRUTH[i]
  local imname = tmpgruth[1]
  local img = image.load(IMREFILE..imname)
  local imgmask = torch.Tensor(480, 640):zero()
  local area = {}
  for k = 1, #tmpgruth[2] do
    local tmptable = tmpgruth[2][k]
		local indleft = tmptable[1]
		local indright = tmptable[2]
		local indtop = tmptable[3]
		local indbottom = tmptable[4]
		local tmparea = (indbottom-indtop+1) * (indright-indleft+1)
		table.insert(area, tmparea)
		imgmask[{ {indtop,indbottom},{indleft,indright} }] = k
  end
  
  for indr = 1, 480, step do
  	if (indr+patchsize-1) > 480 then indr = 480-patchsize+1 end
  	
  	for indc = 1, 640, step do
  	  if (indc+patchsize-1) > 640 then indc = 640-patchsize+1 end
  	  
  	  -- Load image mask 		
  		local tmpmask = imgmask[{ {indr,indr+patchsize-1},{indc,indc+patchsize-1} }]
  		local tmpid = tmpmask:max()
  		local score
  		if tmpid == 0 then
  			score = 0
  		else
  		  for idx = 1, tmpid do
  		    local tmpscore = torch.Tensor(tmpid):zero()
  		    tmpscore[idx] = tmpmask:eq(idx):sum()/area[idx]
  				score = tmpscore:max()
  			end
  		end
  		
  		-- Select data
  		if (score >= 0.7) then
				IMAGEPATCH[{ indpatch,{},{},{} }] = img[{ {},{indr,indr+patchsize-1},{indc,indc+patchsize-1} }] 
				tmpmask = image.scale(tmpmask, masksize, masksize)
				tmpmask[tmpmask:gt(0)] = 1
				IMAGEMASK[{indpatch,{},{} }] = tmpmask
				PATCHSCORE[indpatch] = score
				indpatch = indpatch + 1
				print(indpatch)
  		end
  		
  		if (indc+patchsize-1) == 640 then break end
  	end
  	
  	if (indr+patchsize-1) == 480 then break end
  end
	
end

-- Normalize
local mean, std = normalize_global(IMAGEPATCH)
torch.save('./MEAN_STD.t7', {mean, std})
normalize_local(IMAGEPATCH)

local pos = (1 - PATCHSCORE:eq(0):sum() / totalnum) * 100
print('Positive pencentage: '..pos..'%')

-- Save data
local data = {IMAGEPATCH, IMAGEMASK, PATCHSCORE}
torch.save('./DATA.t7', data)

-- Test data
local ind = 100
image.display(IMAGEPATCH[ind])
image.display(IMAGEMASK[ind])
print(PATCHSCORE[ind])

