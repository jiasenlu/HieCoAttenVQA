--
require 'nn'
require 'nngraph'
require 'misc.LookupTableMaskZero'
require 'cudnn'

local LanguageEmbedding = {}

function LanguageEmbedding.LE(vocab_size, embedding_size, conv_size, seq_length)
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) 

    local seq = inputs[1]

    local embed = nn.Dropout(0.5)(nn.Tanh()(nn.LookupTableMaskZero(vocab_size, embedding_size)(seq)))
    
    table.insert(outputs, embed)
    
    return nn.gModule(inputs, outputs)
end


function LanguageEmbedding.conv(conv_size,embedding_size, seq_length)
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) 

    local embed = inputs[1]

    local unigram = cudnn.TemporalConvolution(embedding_size, conv_size, 1, 1, 0)(embed)
    local bigram = cudnn.TemporalConvolution(embedding_size, conv_size, 2, 1, 1)(embed)
    local trigram = cudnn.TemporalConvolution(embedding_size,conv_size,3, 1, 1)(embed)

    local bigram = nn.Narrow(2,1,seq_length)(bigram)

    local unigram_dim = nn.View(-1, seq_length, conv_size, 1)(unigram)
    local bigram_dim = nn.View(-1, seq_length, conv_size, 1)(bigram)
    local trigram_dim = nn.View(-1, seq_length, conv_size, 1)(trigram)

    local feat = nn.JoinTable(4)({unigram_dim, bigram_dim, trigram_dim})
    local max_feat = nn.Dropout(0.5)(nn.Tanh()(nn.Max(3, 3)(feat)))

    table.insert(outputs, max_feat)

    return nn.gModule(inputs, outputs)
end
--[[
function LanguageEmbedding.conv(conv_size,embedding_size, seq_length)
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) 

    local embed = inputs[1]

    local unigram = nn.TemporalConvolution(embedding_size, conv_size, 1)(embed)
    local bigram = nn.TemporalConvolution(embedding_size, conv_size, 2)(embed)
    local trigram = nn.TemporalConvolution(embedding_size,conv_size,3)(embed)

    local bigram_pad = nn.Padding(1,-1,2,0)(bigram)
    local trigram_pad = nn.Padding(1,1,2,0)(trigram)
    local trigram_pad = nn.Padding(1,-1,2,0)(trigram_pad)

    local unigram_dim = nn.View(seq_length, conv_size, 1):setNumInputDims(3)(unigram)
    local bigram_dim = nn.View(seq_length, conv_size, 1):setNumInputDims(3)(bigram_pad)
    local trigram_dim = nn.View(seq_length, conv_size, 1):setNumInputDims(3)(trigram_pad)

    local feat = nn.JoinTable(4)({unigram_dim, bigram_dim, trigram_dim})
    local max_feat = nn.Dropout(0.5)(nn.Max(3, 3)(feat))

    table.insert(outputs, max_feat)

    return nn.gModule(inputs, outputs)
end
]]--
return LanguageEmbedding
