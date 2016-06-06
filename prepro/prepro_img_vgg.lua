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

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-input_json','../data/vqa_data_prepro.json','path to the json file containing vocab and answers')
cmd:option('-image_root','/home/jiasenlu/data/','path to the image root')
cmd:option('-cnn_proto', '../image_model/VGG_ILSVRC_19_layers_deploy.prototxt', 'path to the cnn prototxt')
cmd:option('-cnn_model', '../image_model/VGG_ILSVRC_19_layers.caffemodel', 'path to the cnn model')

cmd:option('-batch_size', 20, 'batch_size')

cmd:option('-out_name_train', '../data/vqa_data_img_vgg_train.h5', 'output name train')
cmd:option('-out_name_test', '../data/vqa_data_img_vgg_test.h5', 'output name test')

cmd:option('-gpuid', 6, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

--cutorch.setDevice(opt.gpuid)
net=loadcaffe.load(opt.cnn_proto, opt.cnn_model,opt.backend);
print(net)


for i = 46,38,-1 do
    net:remove(i)
end
net:evaluate()
net=net:cuda()
print(net)

imloader={}
function imloader:load(fname)
    self.im=image.load(fname)
end
function loadim(imname)

    imloader:load(imname,  3, 'float')
    im=imloader.im
    dim = im:size(1)
    h = im:size(2)
    w = im:size(3)

    im = image.scale(im, 448, 448)
    -- central crop to 224 * 224
    h = im:size(2)
    w = im:size(3)

    if im:size(1)==1 then
        im2=torch.cat(im,im,1)
        im2=torch.cat(im2,im,1)
        im=im2
    elseif im:size(1)==4 then
        im=im[{{1,3},{},{}}]
    end
    --im = im:sub(1,3, torch.floor(h/2)-224+1, torch.floor(h/2)+224, torch.floor(w/2)-224+1, torch.floor(w/2)+224)
    im=im*255;
    im2=im:clone()
    im2[{{3},{},{}}]=im[{{1},{},{}}]-123.68
    im2[{{2},{},{}}]=im[{{2},{},{}}]-116.779
    im2[{{1},{},{}}]=im[{{3},{},{}}]-103.939

    return im2
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
local feat_train=torch.FloatTensor(sz, 14, 14, 512) --ndims)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    local ims=torch.CudaTensor(r-i+1,3,448,448)
    for j=1,r-i+1 do
        ims[j]=loadim(img_list_train[i+j-1]):cuda()
    end
    net:forward(ims)
    feat_train[{{i,r},{}}]=net.modules[37].output:permute(1,3,4,2):contiguous():float()
    collectgarbage()
end


local ndims=4096
local batch_size = opt.batch_size
local sz=#img_list_test
local feat_test=torch.FloatTensor(sz,14, 14, 512)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    local ims=torch.CudaTensor(r-i+1,3,448,448)
    for j=1,r-i+1 do
        ims[j]=loadim(img_list_test[i+j-1]):cuda()
    end
    net:forward(ims)
    feat_test[{{i,r},{}}]=net.modules[37].output:permute(1,3,4,2):contiguous():float()

    collectgarbage()
end



local train_h5_file = hdf5.open(opt.out_name_train, 'w')
train_h5_file:write('/images_train', feat_train)
train_h5_file:close()

local test_h5_file = hdf5.open(opt.out_name_test, 'w')
test_h5_file:write('/images_test', feat_test)
test_h5_file:close()    


