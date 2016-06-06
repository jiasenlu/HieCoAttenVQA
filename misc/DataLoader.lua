require 'hdf5'
local utils = require 'misc.utils'
local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
    
    if opt.h5_img_file_train ~= nil then
        print('DataLoader loading h5 image file: ', opt.h5_img_file_train)
        local h5_file = hdf5.open(opt.h5_img_file_train, 'r')
        self.fv_im_train = h5_file:read('/images_train'):all()
        h5_file:close()
    end

    if opt.h5_img_file_train ~= nil then    
        print('DataLoader loading h5 image file: ', opt.h5_img_file_test)
        local h5_file = hdf5.open(opt.h5_img_file_test, 'r')
        self.fv_im_test = h5_file:read('/images_test'):all()
        h5_file:close()
    end

    print('DataLoader loading h5 question file: ', opt.h5_ques_file)
    local h5_file = hdf5.open(opt.h5_ques_file, 'r')
    self.ques_train = h5_file:read('/ques_train'):all()
    self.ques_len_train = h5_file:read('/ques_len_train'):all()
    self.img_pos_train = h5_file:read('/img_pos_train'):all()
    self.ques_id_train = h5_file:read('/ques_id_train'):all()
    self.answer = h5_file:read('/answers'):all()
    self.split_train = h5_file:read('/split_train'):all()

    self.ques_test = h5_file:read('/ques_test'):all()
    self.ques_len_test = h5_file:read('/ques_len_test'):all()
    self.img_pos_test = h5_file:read('/img_pos_test'):all()
    self.ques_id_test = h5_file:read('/ques_id_test'):all()
    self.split_test = h5_file:read('/split_test'):all()
    self.ans_test = h5_file:read('/ans_test'):all()

    h5_file:close()
    print('Transform the image feature...')

    if opt.h5_img_file_train ~= nil then
        if opt.feature_type == 'VGG' then
            self.fv_im_train = self.fv_im_train:view(-1, 196, 512):contiguous()
        elseif opt.feature_type == 'Residual' then
            self.fv_im_train = self.fv_im_train:view(-1, 196, 2048):contiguous()
        else
            error('feature type error')
        end
    end

    if opt.h5_img_file_test ~= nil then
        if opt.feature_type == 'VGG' then
            self.fv_im_test = self.fv_im_test:view(-1, 196, 512):contiguous()
        elseif opt.feature_type == 'Residual' then
            self.fv_im_test = self.fv_im_test:view(-1, 196, 2048):contiguous()
        else
            error('feature type error')
        end
    end

    print('DataLoader loading json file: ', opt.json_file)
    local json_file = utils.read_json(opt.json_file)
    self.ix_to_word = json_file.ix_to_word
    self.ix_to_ans = json_file.ix_to_ans

    self.seq_length = self.ques_train:size(2)

    -- count the vocabulary key!
    self.vocab_size = utils.count_key(self.ix_to_word)

    -- Let's get the split for train and val and test.
    self.split_ix = {}
    self.iterators = {}

    for i = 1,self.split_train:size(1) do
        local idx = self.split_train[i]
        if not self.split_ix[idx] then 
            self.split_ix[idx] = {}
            self.iterators[idx] = 1
        end
        table.insert(self.split_ix[idx], i)
    end

    for i = 1,self.split_test:size(1) do
        local idx = self.split_test[i]
        if not self.split_ix[idx] then 
            self.split_ix[idx] = {}
            self.iterators[idx] = 1
        end
        table.insert(self.split_ix[idx], i)
    end

    for k,v in pairs(self.split_ix) do
        print(string.format('assigned %d images to split %s', #v, k))
    end
    collectgarbage() -- do it often and there is no harm ;)
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getVocabSize()
    return self.vocab_size
end

function DataLoader:getSeqLength()
  return self.seq_length
end

function DataLoader:getDataNum(split)
    return #self.split_ix[split]
end

function DataLoader:getBatch(opt)
    local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
    local batch_size = utils.getopt(opt, 'batch_size', 128)

    local split_ix = self.split_ix[split]
    assert(split_ix, 'split ' .. split .. ' not found.')
  
    local max_index = #split_ix
    local infos = {}
    local ques_idx = torch.LongTensor(batch_size):fill(0)
    local img_idx = torch.LongTensor(batch_size):fill(0)

    for i=1,batch_size do
        local ri = self.iterators[split] -- get next index from iterator
        local ri_next = ri + 1 -- increment iterator
        if ri_next > max_index then ri_next = 1 end
        self.iterators[split] = ri_next
        if split == 0 then
            ix = split_ix[torch.random(max_index)]
        else
            ix = split_ix[ri]
        end
        assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)
        ques_idx[i] = ix
        if split == 0 or split == 1 then
            img_idx[i] = self.img_pos_train[ix]
        else
            img_idx[i] = self.img_pos_test[ix]
        end
    end

    local data = {}
    -- fetch the question and image features.
    if split == 0 or split == 1 then
        data.images = self.fv_im_train:index(1,img_idx)
        data.questions = self.ques_train:index(1, ques_idx)
        data.ques_id = self.ques_id_train:index(1, ques_idx)
        data.ques_len = self.ques_len_train:index(1, ques_idx)
        data.answer = self.answer:index(1, ques_idx)
    else
        data.images = self.fv_im_test:index(1,img_idx)
        data.questions = self.ques_test:index(1, ques_idx)
        data.ques_id = self.ques_id_test:index(1, ques_idx)
        data.ques_len = self.ques_len_test:index(1, ques_idx)        
        data.answer = self.ans_test:index(1, ques_idx)        
    end
    return data
end
