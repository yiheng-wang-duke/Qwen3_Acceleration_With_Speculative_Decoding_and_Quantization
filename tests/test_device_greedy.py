import copy
import pytest
import torch
from transformers import Qwen3ForCausalLM

from acceleratellm.decoding.autoregressive import AutoregressiveDecoder
from acceleratellm.decoding.speculative import SpeculativeDecoder
from acceleratellm.models.hf_backend import HFModelAdapter
from toy_models import TableModel


@pytest.mark.parametrize("gamma",[1,2,4,8])
@pytest.mark.parametrize("budget",[0,1,2,5,17])
@pytest.mark.parametrize("eos",[set(),{2}])
@pytest.mark.parametrize("prompt",[[[0]],[[1,2,0]]])
def test_device_greedy_matches_ar_under_rejection_eos_and_budget(gamma,budget,eos,prompt):
    choose=lambda p:(sum((i+1)*x for i,x in enumerate(p))+len(p))%4
    draft_choose=lambda p:(choose(p)+(len(p)%3==0))%4
    target,draft=TableModel(choose=choose),TableModel(choose=draft_choose)
    settings={"strategy":"greedy","max_new_tokens":budget,"temperature":1.,"top_p":1.}
    ids=torch.tensor(prompt)
    expected=AutoregressiveDecoder().generate(target,ids,settings,eos)
    result=SpeculativeDecoder(gamma,greedy_implementation="device").generate(target,ids,settings,eos,draft)
    assert result.token_ids==expected.token_ids
    assert result.stop_reason==expected.stop_reason
    assert 0<=result.accepted_tokens<=result.proposed_tokens
    assert len(result.token_ids)<=budget


@pytest.mark.parametrize("case",["accepted_eos","bonus_eos","rejected_eos","all_accepted"])
def test_device_greedy_all_acceptance_and_eos_cases(case):
    choose=lambda p:1
    target_choose=draft_choose=choose
    if case=="accepted_eos":target_choose=draft_choose=lambda p:2
    elif case=="bonus_eos":target_choose=lambda p:1 if len(p)<5 else 2
    elif case=="rejected_eos":draft_choose=lambda p:2
    target,draft=TableModel(choose=target_choose),TableModel(choose=draft_choose)
    settings={"strategy":"greedy","max_new_tokens":13,"temperature":1.,"top_p":1.}
    ids=torch.tensor([[0]])
    expected=AutoregressiveDecoder().generate(target,ids,settings,{2})
    actual=SpeculativeDecoder(4,greedy_implementation="device").generate(target,ids,settings,{2},draft)
    assert actual.token_ids==expected.token_ids and actual.stop_reason==expected.stop_reason


DEVICES=["cpu"]+(["cuda"] if torch.cuda.is_available() else [])

@pytest.mark.parametrize("device",DEVICES)
@pytest.mark.parametrize("gamma",[2,4,8])
def test_device_greedy_real_transformer_cache_rollback(tiny_model,device,gamma):
    torch.manual_seed(53)
    draft_model=Qwen3ForCausalLM(copy.deepcopy(tiny_model.config)).eval()
    dtype=torch.float32
    target=HFModelAdapter(tiny_model.to(device=device,dtype=dtype))
    draft=HFModelAdapter(draft_model.to(device=device,dtype=dtype))
    ids=torch.tensor([[4,8,6]],device=device)
    settings={"strategy":"greedy","max_new_tokens":17,"temperature":1.,"top_p":1.}
    expected=AutoregressiveDecoder().generate(target,ids,settings,set())
    for validate in (True,False):
        actual=SpeculativeDecoder(gamma,validate_logits=validate,greedy_implementation="device").generate(target,ids,settings,set(),draft)
        assert actual.token_ids==expected.token_ids
