require 'nn'
local utils = require 'misc.utils'
require 'loadcaffe'

--local LSTM_img = require 'misc.LSTM_img'

-------------------------------------------------------------------------------
-- Image Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.cnnModel', 'nn.Module')
function layer:__init(opt)
    parent.__init(self)

    local layer_num = utils.getopt(opt, 'layer_num', 37)
    self.input_size = utils.getopt(opt, 'input_size_image')
    local dropout = utils.getopt(opt, 'dropout', 0)
    self.output_size = utils.getopt(opt, 'output_size')
    self.cnn_proto = utils.getopt(opt, 'cnn_proto')
    self.cnn_model = utils.getopt(opt, 'cnn_model')
    self.backend  = utils.getopt(opt, 'backend')
    -- option for Image Model
    self.h = utils.getopt(opt, 'h')
    self.w = utils.getopt(opt, 'w')
    assert(self.h==self.w) -- h and w should be same here
    self.seq_length = self.h * self.w
    print(self.cnn_proto, self.cnn_model, self.backend)
    local cnn_raw = loadcaffe.load(self.cnn_proto, self.cnn_model, self.backend)
    self.cnn_part = nn.Sequential()
    for i = 1, layer_num do
        local layer = cnn_raw:get(i)
        self.cnn_part:add(layer)
    end
    self.cnn_part:add(nn.View(-1, 512, 196))
    self.cnn_part:add(nn.Transpose({2,3}))
end


function layer:parameters()
    local params = {}
    local grad_params = {}

    local p2,g2 = self.cnn_part:parameters()
    for k,v in pairs(p2) do table.insert(params, v) end
    for k,v in pairs(g2) do table.insert(grad_params, v) end

    return params, grad_params
end

function layer:training()
    self.cnn_part:training()

end

function layer:evaluate()
    self.cnn_part:evaluate()
end

function layer:updateOutput(input)
    local imgs = input
    self.output = self.cnn_part:forward(imgs)

    return self.output

end

function layer:updateGradInput(input, gradOutput)
    local imgs = input

    local dummy = self.cnn_part:backward(imgs, gradOutput)
    self.gradInput = {}
    return self.gradInput
end


