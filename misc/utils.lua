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


function utils.prepro(im, on_gpu)
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  im=im*255
  local im2=im:clone()
  im2[{{},{3},{},{}}]=im[{{},{1},{},{}}]-123.68
  im2[{{},{2},{},{}}]=im[{{},{2},{},{}}]-116.779
  im2[{{},{1},{},{}}]=im[{{},{3},{},{}}]-103.939  
  
  return im2
end


return utils