require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

dofile('demo_createlabel.lua')
model = torch.load('./trained_models/model.t7'):cuda()
model:evaluate()

local imname = '00001.ppm'
local img = image.load(IMFILE..imname)
img = image.scale(img, 640, 480)
local output = model:forward(img:cuda())
output = output[1]:double()

local _, indCol = output[1]:max(1):max(1)
local _, indRow = output[1]:max(2):max(1)

indCol = indCol[1][1]
indRow = indRow[1][1]

print(indCol)
print(indRow)

local disCol = output[{ 2,indRow,indCol }]
local disRow = output[{ 3,indRow,indCol }]
local W = output[{ 4,indRow,indCol }]
local H = output[{ 5,indRow,indCol }]
-- Generate box
local Col = torch.round((indCol - 1) * 4 - disCol * 640)
local Row = torch.round((indRow - 1) * 4 - disRow * 480)
W = torch.round(W * 640)
H = torch.round(H * 480)
print(Row..', '..Col..', '..W..', '..H)
draw.drawBox(img, Row, Col, W, H, 4)
image.display(img)



