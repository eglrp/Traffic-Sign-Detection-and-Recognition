require 'torch'
require 'image'
draw = require 'draw'

GTFILE = '/media/administrator/文档/我的文件/Tencent_Model/GTSDB/FullIJCNN2013/gt.txt'
IMFILE = '/media/administrator/文档/我的文件/Tencent_Model/GTSDB/FullIJCNN2013/'

GTRUTH = {}

local f = assert(io.open(GTFILE, 'r'))

local id = 1
for i = 1, 10 do
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
  local cencol = torch.round(leftcol + (rightcol - leftcol) / 2)
  local cenrow = torch.round(toprow + (bottomrow - toprow) / 2)
  local indCol = torch.ceil(cencol / 4)
  local indRow = torch.ceil(cenrow / 4)
  
  -- Normalized ground truth
  disCol = ((indCol - 1) * 4 - leftcol) / 640
  disRow = ((indRow - 1) * 4 - toprow) / 480
  W = (rightcol - leftcol + 1) / 640
  H = (bottomrow - toprow + 1) /  480
  indCol = torch.ceil(cencol / 4) / 160
  indRow = torch.ceil(cenrow / 4) / 120
  
  table.insert(GTRUTH, {imname, indCol, indRow, disCol, disRow, W, H})
  print(id)
  id = id + 1
end
--print(GTRUTH)

-- Test labels, back to rectangle, we know indX, indY, disX, disY, W, H
--[[
local imname = '00001.ppm'
local img = image.load(IMFILE..imname)
img = image.scale(img, 640, 480)
for i = 1, #GTRUTH do
	local tmp = GTRUTH[i]
	if (tmp[1] == imname) then
		local indCol = tmp[2]
		local indRow = tmp[3]
		local disCol = tmp[4]
		local disRow = tmp[5]
		local W = tmp[6]
		local H = tmp[7]
		-- Generate box
		local Col = torch.round((indCol * 160 - 1) * 4 - disCol * 640)
    local Row = torch.round((indRow * 120 - 1) * 4 - disRow * 480)
		W = torch.round(W * 640)
		H = torch.round(H * 480)
		print(Row..', '..Col..', '..W..', '..H)
		draw.drawBox(img, Row, Col, W, H, 4)
	end
end
image.display(img)
--]]

