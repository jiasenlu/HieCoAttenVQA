require 'nn'
local utils = require 'misc.utils'
local attention = require 'misc.attention'

local layer, parent = torch.class('nn.recursive_atten', 'nn.Module')
function layer:__init(opt)
    parent.__init(self)

    self.atten_encode = attention.recursive_atten(512,512,1024,1000)

    -- self.atten_encode = attention.recursive_atten(512,512,512,1000) -- coco_qa
end

function layer:getModulesList()
    return {self.atten_encode}
end

function layer:parameters()
    local p1,g1 = self.atten_encode:parameters()
    local params = {}
    for k,v in pairs(p1) do table.insert(params, v) end
    local grad_params = {}
    for k,v in pairs(g1) do table.insert(grad_params, v) end

    return params, grad_params
end

function layer:training()
    self.atten_encode:training()
end

function layer:evaluate()
    self.atten_encode:evaluate()
end

function layer:updateOutput(input)
  local out_feat = self.atten_encode:forward(input)
  return out_feat
end

function layer:updateGradInput(input, gradOutput)
  self.gradInput = self.atten_encode:backward(input, gradOutput)
  return self.gradInput
end
