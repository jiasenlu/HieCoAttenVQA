local cjson = require 'cjson'
local utils = {}
require 'nn'
-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end

  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  --cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

function utils.right_align(seq, lengths)
    -- right align the questions. 
    local v=seq:clone():fill(0)
    local N=seq:size(2)
    for i=1,seq:size(1) do
        v[i][{{N-lengths[i]+1,N}}]=seq[i][{{1,lengths[i]}}]
    end
    return v
end

function utils.normlize_image(imgFeat)
    local length = imgFeat:size(2)
    local nm=torch.sqrt(torch.sum(torch.cmul(imgFeat,imgFeat),2)) 
    return torch.cdiv(imgFeat,torch.repeatTensor(nm,1,length)):float()
end

function utils.count_key(t)
    local count = 1
    for i, w in pairs(t) do 
        count = count + 1 
    end
    return count
end


function utils.prepro(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = 448
  --[[
  -- cropping data augmentation, if needed
  if h > cnn_input_size or w > cnn_input_size then 
    local xoff, yoff
    if data_augment then
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end
  ]]--
  -- ship to gpu or convert from byte to float
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end
  -- lazily instantiate vgg_mean

  if not utils.vgg_mean then
  utils.vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}:view(1,3,1,1) --BRG
  --utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1)
  end
  utils.vgg_mean = utils.vgg_mean:typeAs(imgs) -- a noop if the types match

  -- subtract vgg mean
  imgs:add(-1, utils.vgg_mean:expandAs(imgs))
  
  return imgs
end

-- layer that expands features out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.FeatExpander', 'nn.Module')
function layer:__init(n)
  parent.__init(self)
  self.n = n
end
function layer:updateOutput(input)
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency
  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 3)
  local d = input:size(2)
  local z = input:size(3)
  self.output:resize(input:size(1)*self.n, d, z)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{{j,j+self.n-1}}] = input[{ {k,k}, {}, {} }]:expand(self.n, d, z) -- copy over
  end
  return self.output
end
function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)

  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end

return utils