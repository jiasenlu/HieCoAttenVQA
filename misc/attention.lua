require 'nngraph'
require 'nn'
require 'misc.maskSoftmax'
local attention = {}
function attention.parallel_atten(input_size_ques, input_size_img, embedding_size, ques_seq_size, img_seq_size)
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 

    local ques_feat = inputs[1]
    local img_feat = inputs[2]
    local mask = inputs[3]


    local img_corr_dim = nn.Linear(input_size_img, input_size_ques)(nn.View(input_size_img):setNumInputDims(2)(img_feat))
    local img_corr = nn.View(img_seq_size, embedding_size):setNumInputDims(2)(img_corr_dim)

    local weight_matrix = nn.Tanh()(nn.MM(false, true)({ques_feat, img_corr}))

    local ques_embed_dim = nn.Linear(input_size_ques, embedding_size)(nn.View(input_size_ques):setNumInputDims(2)(ques_feat))
    local ques_embed = nn.View(ques_seq_size, embedding_size):setNumInputDims(2)(ques_embed_dim)

    local img_embed_dim = nn.Linear(input_size_img, input_size_ques)(nn.View(input_size_img):setNumInputDims(2)(img_feat))
    local img_embed = nn.View(img_seq_size, embedding_size):setNumInputDims(2)(img_embed_dim)

    local transform_img = nn.MM(false, false)({weight_matrix, img_embed})
    local ques_atten_sum = nn.Dropout(0.5)(nn.Tanh()(nn.CAddTable()({transform_img, ques_embed})))
    local ques_atten_embedding = nn.Linear(embedding_size, 1)(nn.View(embedding_size):setNumInputDims(2)(ques_atten_sum))
    local ques_atten = nn.maskSoftMax()({nn.View(ques_seq_size):setNumInputDims(2)(ques_atten_embedding),mask})

    local transform_ques = nn.MM(true, false)({weight_matrix, ques_embed})
    local img_atten_sum = nn.Dropout(0.5)(nn.Tanh()(nn.CAddTable()({transform_ques, img_embed})))
    local img_atten_embedding = nn.Linear(embedding_size, 1)(nn.View(embedding_size):setNumInputDims(2)(img_atten_sum))
    local img_atten = nn.SoftMax()(nn.View(img_seq_size):setNumInputDims(2)(img_atten_embedding))

    local ques_atten_dim = nn.View(1,-1):setNumInputDims(1)(ques_atten)
    local img_atten_dim = nn.View(1,-1):setNumInputDims(1)(img_atten)

    local ques_atten_feat = nn.MM(false, false)({ques_atten_dim, ques_feat})
    local ques_atten_feat = nn.View(input_size_ques):setNumInputDims(2)(ques_atten_feat)

    local img_atten_feat = nn.MM(false, false)({img_atten_dim, img_feat})
    local img_atten_feat = nn.View(input_size_img):setNumInputDims(2)(img_atten_feat)

    table.insert(outputs, ques_atten_feat)
    table.insert(outputs, img_atten_feat)

    return nn.gModule(inputs, outputs)
end


function attention.alternating_atten(input_size_ques, input_size_img, embedding_size, ques_seq_size, img_seq_size)
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 

    local ques_feat = inputs[1]
    local img_feat = inputs[2]
    local mask = inputs[3]

    local ques_embed_dim = nn.Linear(input_size_ques, embedding_size)(nn.View(-1, input_size_ques)(ques_feat))
    local ques_embed = nn.View(-1, ques_seq_size, embedding_size)(ques_embed_dim)
    
    local feat = nn.Dropout(0.5)(nn.Tanh()(ques_embed))
    local h1 = nn.Linear(embedding_size, 1)(nn.View(-1, embedding_size)(feat))
    local P1 = nn.maskSoftMax()({nn.View(-1, ques_seq_size)(h1),mask})
    local ques_atten = nn.View(1,-1):setNumInputDims(1)(P1)
    local quesAtt1 = nn.MM(false, false)({ques_atten, ques_feat})
    local ques_atten_feat_1 = nn.View(-1, input_size_ques)(quesAtt1)


    -- img attention
    local ques_embed_img = nn.Linear(input_size_ques, embedding_size)(ques_atten_feat_1)

    local img_embed_dim = nn.Linear(input_size_img, embedding_size)(nn.View(-1, input_size_img)(img_feat))   

    local img_embed = nn.View(-1, img_seq_size, embedding_size)(img_embed_dim)

    local ques_replicate = nn.Replicate(img_seq_size,2)(ques_embed_img)
    
    local feat = nn.Dropout(0.5)(nn.Tanh()(nn.CAddTable()({img_embed, ques_replicate})))
    local h2 = nn.Linear(embedding_size, 1)(nn.View(-1, embedding_size)(feat))
    local P2 = nn.SoftMax()(nn.View(-1, img_seq_size)(h2))
    local img_atten = nn.View(1,-1):setNumInputDims(1)(P2)
    local visAtt = nn.MM(false, false)({img_atten, img_feat})
    local img_atten_feat = nn.View(-1, input_size_img)(visAtt)

    -- question attention

    local img_embed = nn.Linear(input_size_img, embedding_size)(img_atten_feat)
    local img_replicate = nn.Replicate(ques_seq_size,2)(img_embed)

    local ques_embed_dim = nn.Linear(input_size_ques, embedding_size)(nn.View(-1, input_size_ques)(ques_feat))
    local ques_embed = nn.View(-1, ques_seq_size, embedding_size)(ques_embed_dim)

    local feat = nn.Dropout(0.5)(nn.Tanh()(nn.CAddTable()({ques_embed, img_replicate})))

    local h3 = nn.Linear(embedding_size, 1)(nn.View(-1, embedding_size)(feat))
    local P3 = nn.maskSoftMax()({nn.View(-1, ques_seq_size)(h3),mask})
    local probs3dim = nn.View(1,-1):setNumInputDims(1)(P3)
    local quesAtt = nn.MM(false, false)({probs3dim, ques_feat})
    local ques_atten_feat = nn.View(-1, 512)(quesAtt)
    
    -- combine image attention feature and language attention feature

    table.insert(outputs, ques_atten_feat)
    table.insert(outputs, img_atten_feat)
    
    --table.insert(outputs, probs3dim)
    ---table.insert(outputs, img_atten)
    
    return nn.gModule(inputs, outputs)
end

function attention.fuse(input_size, ques_seq_size)
    local inputs = {}
    local outputs = {}    
    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()())     

    local fore_lstm = inputs[1]
    local back_lstm = inputs[2]

    local concat_lstm = nn.JoinTable(2)({fore_lstm, back_lstm})

    local ques_feat_dim = nn.Linear(input_size*2, input_size)(nn.View(input_size*2):setNumInputDims(2)(concat_lstm))
    local ques_feat = nn.Dropout(0.5)(nn.View(ques_seq_size, input_size):setNumInputDims(2)(ques_feat_dim))

    table.insert(outputs, ques_feat)
    return nn.gModule(inputs, outputs)    
end


function attention.recursive_atten(input_size, embedding_size, last_embed_size, output_size)
    local inputs = {}
    local outputs = {}
    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 
    table.insert(inputs, nn.Identity()()) 

    local embed_ques = inputs[1]
    local embed_img = inputs[2]
    local conv_ques = inputs[3]
    local conv_img = inputs[4]
    local lstm_ques = inputs[5]
    local lstm_img = inputs[6]

    local feat1 = nn.Dropout(0.5)(nn.CAddTable()({embed_ques, embed_img}))
    local hidden1 = nn.Tanh()(nn.Linear(input_size, embedding_size)(feat1))
    local feat2 = nn.Dropout(0.5)(nn.JoinTable(2)({nn.CAddTable()({conv_ques, conv_img}), hidden1}))
    local hidden2 = nn.Tanh()(nn.Linear(embedding_size+input_size, embedding_size)(feat2))
    local feat3 = nn.Dropout(0.5)(nn.JoinTable(2)({nn.CAddTable()({lstm_ques, lstm_img}), hidden2}))
    local hidden3 = nn.Tanh()(nn.Linear(embedding_size+input_size, last_embed_size)(feat3))
    local outfeat = nn.Linear(last_embed_size, output_size)(nn.Dropout(0.5)(hidden3))

    table.insert(outputs, outfeat)

    return nn.gModule(inputs, outputs)
end

return attention
