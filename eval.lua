
require 'nn'
require 'torch'
require 'optim'
require 'misc.DataLoader'
require 'misc.word_level'
require 'misc.phrase_level'
require 'misc.ques_level'
require 'misc.recursive_atten'
require 'misc.optim_updates'
local utils = require 'misc.utils'
require 'xlua'


cmd = torch.CmdLine()
cmd:text()
cmd:text('evaluate a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_train_h5','data/vqa_data_img_vgg_train.h5','path to the h5file containing the image feature')
cmd:option('-input_img_test_h5','data/vqa_data_img_vgg_test.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data/vqa_data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/vqa_data_prepro.json','path to the json file containing additional info and vocab')

cmd:option('-start_from', 'model/vqa_model/model_alternating_train_vgg.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-co_atten_type', 'Alternating', 'co_attention type. Parallel or Alternating, alternating trains more faster than parallel.')
cmd:option('-feature_type', 'VGG', 'VGG or Residual')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 2, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')

cmd:text()

local batch_size = 256

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then 
  require 'cudnn' 
  end
  cutorch.manualSeed(opt.seed)
  --cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
end

opt = cmd:parse(arg)

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
local protos = {}
print('Building the model...')
-- intialize language model
local loaded_checkpoint
local lmOpt
if string.len(opt.start_from) > 0 then

  loaded_checkpoint = torch.load(opt.start_from)
  lmOpt = loaded_checkpoint.lmOpt
else
  lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = opt.rnn_layers
  lmOpt.dropout = 0.5
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size
  lmOpt.output_size = opt.rnn_size
  lmOpt.atten_type = opt.co_atten_type
  lmOpt.feature_type = opt.feature_type
end
lmOpt.hidden_size = 512
lmOpt.feature_type = 'VGG'
lmOpt.atten_type = opt.co_atten_type
print(lmOpt)

protos.word = nn.word_level(lmOpt)
protos.phrase = nn.phrase_level(lmOpt)
protos.ques = nn.ques_level(lmOpt)

protos.atten = nn.recursive_atten()
protos.crit = nn.CrossEntropyCriterion()

if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

local wparams, grad_wparams = protos.word:getParameters()
local pparams, grad_pparams = protos.phrase:getParameters()
local qparams, grad_qparams = protos.ques:getParameters()
local aparams, grad_aparams = protos.atten:getParameters()


if string.len(opt.start_from) > 0 then
  print('Load the weight...')
  wparams:copy(loaded_checkpoint.wparams)
  pparams:copy(loaded_checkpoint.pparams)
  qparams:copy(loaded_checkpoint.qparams)
  aparams:copy(loaded_checkpoint.aparams)
end

print('total number of parameters in word_level: ', wparams:nElement())
assert(wparams:nElement() == grad_wparams:nElement())

print('total number of parameters in phrase_level: ', pparams:nElement())
assert(pparams:nElement() == grad_pparams:nElement())

print('total number of parameters in ques_level: ', qparams:nElement())
assert(qparams:nElement() == grad_qparams:nElement())
protos.ques:shareClones()

print('total number of parameters in recursive_attention: ', aparams:nElement())
assert(aparams:nElement() == grad_aparams:nElement())

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------

local loader = DataLoader{h5_img_file_train = opt.input_img_train_h5, h5_img_file_test = opt.input_img_test_h5, h5_ques_file = opt.input_ques_h5, json_file = opt.input_json, feature_type = opt.feature_type}

collectgarbage() 

function eval_split(split)

  protos.word:evaluate()
  protos.phrase:evaluate()
  protos.ques:evaluate()
  protos.atten:evaluate()
  loader:resetIterator(split)

  local n = 0
  local loss_evals = 0  
  local predictions = {}
  local total_num = loader:getDataNum(2)
  print(total_num)
  local logprob_all = torch.Tensor(total_num, 1000)
  local ques_id = torch.Tensor(total_num)

  for i = 1, total_num, batch_size do
    xlua.progress(i, total_num)
    local r = math.min(i+batch_size-1, total_num) 

    local data = loader:getBatch{batch_size = r-i+1, split = split}
    -- ship the data to cuda
    if opt.gpuid >= 0 then
      data.images = data.images:cuda()
      data.questions = data.questions:cuda()
      data.ques_len = data.ques_len:cuda()
    end

    local word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward({data.questions, data.images}))

    local conv_feat, p_ques, p_img = unpack(protos.phrase:forward({word_feat, data.ques_len, img_feat, mask}))

    local q_ques, q_img = unpack(protos.ques:forward({conv_feat, data.ques_len, img_feat, mask}))

    local feature_ensemble = {w_ques, w_img, p_ques, p_img, q_ques, q_img}
    local out_feat = protos.atten:forward(feature_ensemble)

    logprob_all:sub(i, r):copy(out_feat:float())
    ques_id:sub(i, r):copy(data.ques_id)

    end


    tmp,pred=torch.max(logprob_all,2);

    for i=1,total_num do
        local ans = loader.ix_to_ans[tostring(pred[{i,1}])]
        table.insert(predictions,{question_id=ques_id[i],answer=ans})
    end

  return {predictions}
end

predictions = eval_split(2)

utils.write_json('OpenEnded_mscoco_co-atten_results.json', predictions[1])

--utils.write_json('MultipleChoice_mscoco_co-atten_results.json', predictions[2])
