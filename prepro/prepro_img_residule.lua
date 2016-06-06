require 'nn'
require 'optim'
require 'torch'
require 'math'
require 'cunn'
require 'cutorch'
require 'loadcaffe'
require 'image'
require 'hdf5'
cjson=require('cjson') 
require 'xlua'
require 'cudnn'
local t = require 'image_model.transforms'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-input_json','../data/vqa_data_prepro.json','path to the json file containing vocab and answers')
cmd:option('-image_root','/home/jiasenlu/data/','path to the image root')

cmd:option('-residule_path', '')
cmd:option('-batch_size', 10, 'batch_size')

cmd:option('-out_name_train', 'data/cocoqa_data_img_residule_train.h5', 'output name')
cmd:option('-out_name_test', 'data/cocoqa_data_img_residule_test.h5', 'output name')

cmd:option('-gpuid', 6, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

--cutorch.setDevice(opt.gpuid)
local model = torch.load(opt.residule_path)


for i = 14,12,-1 do
    model:remove(i)
end
print(model)
model:evaluate()
model=model:cuda()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.ColorNormalize(meanstd)
}

imloader={}
function imloader:load(fname)
    self.im=image.load(fname)
end
function loadim(imname)

    imloader:load(imname,  3, 'float')
    im=imloader.im
    im = image.scale(im, 448, 448)

    if im:size(1)==1 then
        im2=torch.cat(im,im,1)
        im2=torch.cat(im2,im,1)
        im=im2
    elseif im:size(1)==4 then
        im=im[{{1,3},{},{}}]
    end

    im = transform(im)
    return im
end

local image_root = opt.image_root
-- open the mdf5 file

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local img_list_train={}
for i,imname in pairs(json_file['unique_img_train']) do
    table.insert(img_list_train, image_root .. imname)
end

local img_list_test={}
for i,imname in pairs(json_file['uniuqe_img_test']) do
    table.insert(img_list_test, image_root .. imname)
end

local ndims=4096
local batch_size = opt.batch_size
local sz=#img_list_train
local feat_train=torch.FloatTensor(sz, 14, 14, 2048) --ndims)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    local ims=torch.CudaTensor(r-i+1,3,448,448)
    for j=1,r-i+1 do
        ims[j]=loadim(img_list_train[i+j-1]):cuda()
    end
    local output = model:forward(ims)
    feat_train[{{i,r},{}}]=output:permute(1,3,4,2):contiguous():float()
    collectgarbage()
end

local train_h5_file = hdf5.open(opt.out_name_train, 'w')
train_h5_file:write('/images_train', feat_train)
train_h5_file:close()


local ndims=4096
local batch_size = opt.batch_size
local sz=#img_list_test
local feat_test=torch.FloatTensor(sz,14, 14, 2048)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    local ims=torch.CudaTensor(r-i+1,3,448,448)
    for j=1,r-i+1 do
        ims[j]=loadim(img_list_test[i+j-1]):cuda()
    end
    local output = model:forward(ims)
    feat_test[{{i,r},{}}]=output:permute(1,3,4,2):contiguous():float()
    collectgarbage()
end

local test_h5_file = hdf5.open(opt.out_name_test, 'w')
test_h5_file:write('/images_test', feat_test)
test_h5_file:close()    
