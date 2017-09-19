-- Loads the merged p(e|m) index.
if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end


require 'torch'
tds = tds or require 'tds' 

dofile 'utils/utils.lua'
dofile 'entities/ent_name2id_freq/ent_name_id.lua'

ent_p_e_m_index = tds.Hash()

mention_lower_to_one_upper = tds.Hash()

mention_total_freq = tds.Hash()

local crosswikis_textfilename = opt.root_data_dir .. 'generated/crosswikis_wikipedia_p_e_m.txt'
print('==> Loading crosswikis_wikipedia from file ' .. crosswikis_textfilename)
local it, _ = io.open(crosswikis_textfilename)
local line = it:read()

local num_lines = 0
while (line) do
  num_lines = num_lines + 1
  if num_lines % 2000000 == 0 then
    print('Processed ' .. num_lines .. ' lines. ')
  end

  local parts = split(line , '\t')
  local mention = parts[1]

  local total = tonumber(parts[2])
  assert(total)
  if total >= 1 then
    ent_p_e_m_index[mention] = tds.Hash()
    mention_lower_to_one_upper[mention:lower()] = mention
    mention_total_freq[mention] = total
    local num_parts = table_len(parts)
    for i = 3, num_parts do
      local ent_str = split(parts[i], ',')
      local ent_wikiid = tonumber(ent_str[1])
      local freq = tonumber(ent_str[2])
      assert(ent_wikiid)
      assert(freq)
      ent_p_e_m_index[mention][ent_wikiid] = freq / (total + 0.0) -- not sorted
    end
  end
  line = it:read()
end

local yago_textfilename = opt.root_data_dir .. 'generated/yago_p_e_m.txt'
print('==> Loading yago index from file ' .. yago_textfilename)
it, _ = io.open(yago_textfilename)
line = it:read()

num_lines = 0
while (line) do
  num_lines = num_lines + 1
  if num_lines % 2000000 == 0 then
    print('Processed ' .. num_lines .. ' lines. ')
  end

  local parts = split(line , '\t')
  local mention = parts[1]

  local total = tonumber(parts[2])
  assert(total)
  if total >= 1 then
    mention_lower_to_one_upper[mention:lower()] = mention
    if not mention_total_freq[mention] then
      mention_total_freq[mention] = total
    else
      mention_total_freq[mention] = total + mention_total_freq[mention]
    end
    
    local yago_ment_ent_idx = tds.Hash()
    local num_parts = table_len(parts)
    for i = 3, num_parts do
      local ent_str = split(parts[i], ',')
      local ent_wikiid = tonumber(ent_str[1])
      local freq = 1
      assert(ent_wikiid)
      yago_ment_ent_idx[ent_wikiid] = freq / (total + 0.0) -- not sorted
    end
  
    if not ent_p_e_m_index[mention] then
      ent_p_e_m_index[mention] = yago_ment_ent_idx
    else
      for ent_wikiid,prob in pairs(yago_ment_ent_idx) do
        if not ent_p_e_m_index[mention][ent_wikiid] then
          ent_p_e_m_index[mention][ent_wikiid] = 0.0
        end
        ent_p_e_m_index[mention][ent_wikiid] = math.min(1.0, ent_p_e_m_index[mention][ent_wikiid] + prob)
      end
      
    end
  
  end
  line = it:read()
end

assert(ent_p_e_m_index['Dejan Koturovic'] and ent_p_e_m_index['Jose Luis Caminero'])

-- Function used to preprocess a given mention such that it has higher 
-- chance to have at least one valid entry in the p(e|m) index.
function preprocess_mention(m)
  assert(ent_p_e_m_index and mention_total_freq)
  local cur_m = modify_uppercase_phrase(m)
  if (not ent_p_e_m_index[cur_m]) then
    cur_m = m
  end
  if (mention_total_freq[m] and (mention_total_freq[m] > mention_total_freq[cur_m])) then
    cur_m = m -- Cases like 'U.S.' are handed badly by modify_uppercase_phrase
  end
  -- If we cannot find the exact mention in our index, we try our luck to find it in a case insensitive index.
  if (not ent_p_e_m_index[cur_m]) and mention_lower_to_one_upper[cur_m:lower()] then
    cur_m = mention_lower_to_one_upper[cur_m:lower()]
  end
  return cur_m
end


print('    Done loading index')
