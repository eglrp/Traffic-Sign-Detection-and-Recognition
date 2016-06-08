require 'torch'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

GTFILE = '/media/administrator/文档/我的文件/Tencent_Model/GTSDB/FullIJCNN2013/gt.txt'
IMFILE = '/media/administrator/文档/我的文件/Tencent_Model/GTSDB/FullIJCNN2013/'
SVFILE = '/media/administrator/文档/我的文件/Tencent_Model/GTSDB/resized/'

local f = assert(io.open(GTFILE, 'r'))
local IMGMEAN = torch.Tensor(3, 480, 640):zero()
local id = 1
local img

while(1) do
  local line = f:read("*line")
  if (line == nil) then
    break
  end 
  
  -- Get img name
  local gtruth = string.split(line, ';')
  local imname = gtruth[1]
  
  -- Get image size
  img = image.load(IMFILE..imname)
  img = image.scale(img, 640, 480)
  image.save(SVFILE..imname, img)
  IMGMEAN = IMGMEAN + img
  print(id)
  id = id + 1
end

IMGMEAN = IMGMEAN / id
local imtmp = IMGMEAN:clone()
imtmp[{ 1,{},{} }] = IMGMEAN[{ 3,{},{} }]
imtmp[{ 3,{},{} }] = IMGMEAN[{ 1,{},{} }]
IMGMEAN = imtmp
print(IMGMEAN:max())

-- Save model
torch.save('./trained_models/IMGMEAN.t7', IMGMEAN)

