local maskSoftMax, _ = torch.class('nn.maskSoftMax', 'nn.Module')

function maskSoftMax:updateOutput(input)
   local data = input[1]
   local mask = input[2]
   if(mask:type() == 'torch.CudaTensor') then
      mask = mask:cudaByte()
   end
   
   data:maskedFill(mask, -9999999)
   if(mask:type() == 'torch.CudaByteTensor') then
      mask = mask:cuda()
   end   
   data.THNN.SoftMax_updateOutput(
      data:cdata(),
      self.output:cdata()
   )
   return self.output
end

function maskSoftMax:updateGradInput(input, gradOutput)
   local data = input[1]
   local mask = input[2]
   if(mask:type() == 'torch.CudaTensor') then
      mask = mask:cudaByte()
   end
   
   data:maskedFill(mask, -9999999)
   if(mask:type() == 'torch.CudaByteTensor') then
      mask = mask:cuda()
   end   
   
   data.THNN.SoftMax_updateGradInput(
      data:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   if not self.dummy_out then
      self.dummy_out = mask:clone()
   end
   self.dummy_out:resizeAs(mask):zero()
   return {self.gradInput, self.dummy_out}
end
