require 'torch'
require 'image'
draw = require 'draw'

GTFILE = '/media/administrator/文档/我的文件/Tencent_Model/GTSDB/FullIJCNN2013/gt.txt'
IMFILE = '/media/administrator/文档/我的文件/Tencent_Model/GTSDB/FullIJCNN2013/'

GTRUTH = {}

local f = assert(io.open(GTFILE, 'r'))

local id = 1
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
  local img = image.load(IMFILE..imname)
  local imsize = img:size()
  local imrow = imsize[2]
  local imcol = imsize[3]
  leftcol = leftcol / imcol * 640
  rightcol = rightcol / imcol * 640
  toprow = toprow / imrow * 480
  bottomrow = bottomrow / imrow * 480
  local cenx = torch.round(leftcol + (rightcol - leftcol) / 2)
  local ceny = torch.round(toprow + (bottomrow - toprow) / 2)
  local indX = torch.ceil(cenx / 4)
  local indY = torch.ceil(ceny / 4)
  
  -- Normalized ground truth
  disX = ((indX - 1) * 4 - leftcol) / 640
  disY = ((indY - 1) * 4 - toprow) / 480
  W = (rightcol - leftcol + 1) / 640
  H = (bottomrow - toprow + 1) /  480
  indX = torch.ceil(cenx / 4) / 160
  indY = torch.ceil(ceny / 4) / 120
  
  table.insert(GTRUTH, {imname, indX, indY, disX, disY, W, H})
  print(id)
  id = id + 1
end
print(GTRUTH)

-- Test labels, back to rectangle, we know indX, indY, disX, disY, W, H
--[[
local imname = '00005.ppm'
local img = image.load(IMFILE..imname)
img = image.scale(img, 640, 480)
for i = 1, #GTRUTH do
	local tmp = GTRUTH[i]
	if (tmp[1] == imname) then
		local indX = tmp[2]
		local indY = tmp[3]
		local disX = tmp[4]
		local disY = tmp[5]
		local W = tmp[6]
		local H = tmp[7]
		-- Generate box
		local X = torch.round((indX * 160 - 1) * 4 - disX * 640)
    local Y = torch.round((indY * 120 - 1) * 4 - disY * 480)
		W = torch.round(W * 640)
		H = torch.round(H * 480)
		print(X..', '..Y..', '..W..', '..H)
		draw.drawBox(img, Y, X, W, H, 4)
	end
end
image.display(img)
--]]

