import copy
import heapq
import torch

from .parameters import *

def greedy_search(model, e_output, e_mask, trg_sp, device, return_attn=False):
    last_words = torch.LongTensor([pad_id] * trg_seq_len).to(device) # (L)
    last_words[0] = sos_id # (L)
    cur_len = 1

    model.eval()
    for i in range(trg_seq_len):
        d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
        nopeak_mask = torch.ones([1, trg_seq_len, trg_seq_len], dtype=torch.bool).to(device)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

        trg_embedded = model.trg_embedding(last_words.unsqueeze(0))
        trg_positional_encoded = model.trg_positional_encoder(trg_embedded)
        decoder_output, attn = model.decoder(
            trg_positional_encoded,
            e_output,
            e_mask,
            d_mask
        ) # (1, L, d_model)

        output = model.softmax(
            model.output_linear(decoder_output)
        ) # (1, L, trg_vocab_size)

        output = torch.argmax(output, dim=-1) # (1, L)
        last_word_id = output[0][i].item()

        if i < trg_seq_len-1:
            last_words[i+1] = last_word_id
            cur_len += 1

        if last_word_id == eos_id:
            break

    if last_words[-1].item() == pad_id:
        decoded_output = last_words[1:cur_len].tolist()
    else:
        decoded_output = last_words[1:].tolist()
    decoded_output = trg_sp.decode_ids(decoded_output)
    if return_attn:
        return decoded_output, attn
    else:
        return decoded_output

def beam_search(model, e_output, e_mask, trg_sp, device, return_candidates=False, return_attn=False):
    cur_queue = PriorityQueue()
    #for k in range(beam_size):
    cur_queue.put(BeamNode(sos_id, -0.0, [sos_id]))

    finished_count = 0

    model.eval()
    for pos in range(trg_seq_len):
        new_queue = PriorityQueue()
        for k in range(beam_size):
            if pos ==0 and (k)>0:
                continue
            else:
                node = cur_queue.get()

            node = cur_queue.get()
            if node.is_finished:
                new_queue.put(node)
            else:
                trg_input = torch.LongTensor(node.decoded + [pad_id] * (trg_seq_len - len(node.decoded))).to(device) # (L)
                d_mask = (trg_input.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
                nopeak_mask = torch.ones([1, trg_seq_len, trg_seq_len], dtype=torch.bool).to(device)
                nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
                d_mask = d_mask & nopeak_mask # (1, L, L) padding false

                trg_embedded = model.trg_embedding(trg_input.unsqueeze(0))
                trg_positional_encoded = model.trg_positional_encoder(trg_embedded)
                decoder_output, attn = model.decoder(
                    trg_positional_encoded,
                    e_output,
                    e_mask,
                    d_mask
                ) # (1, L, d_model)

                output = model.softmax(
                    model.output_linear(decoder_output)
                ) # (1, L, trg_vocab_size)

                output = torch.topk(output[0][pos], dim=-1, k=beam_size)
                last_word_ids = output.indices.tolist() # (k)
                last_word_prob = output.values.tolist() # (k)

                for i, idx in enumerate(last_word_ids):
                    new_node = BeamNode(idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx])
                    if idx == eos_id:
                        #new_node.prob = new_node.prob / float(len(new_node.decoded))
                        new_node.is_finished = True
                        finished_count += 1
                    new_queue.put(new_node)

        cur_queue = copy.deepcopy(new_queue)

        #if finished_count == beam_size:
        #    break

    if not return_candidates:
        decoded_output = cur_queue.get().decoded

        if decoded_output[-1] == eos_id:
            decoded_output = decoded_output[1:-1]
        else:
            decoded_output = decoded_output[1:]

        if return_attn:
            return trg_sp.decode_ids(decoded_output), attn
        else:
            return trg_sp.decode_ids(decoded_output)

    else:
        all_candidates = list()
        scores  = [ ]
        for _ in range(beam_size):
            node = cur_queue.get()
            decoded_output = node.decoded
            scores.append(node.prob)
            if decoded_output[-1] == eos_id:
                decoded_output = decoded_output[1:-1]
            else:
                decoded_output = decoded_output[1:]
            all_candidates.append(trg_sp.decode_ids(decoded_output))

        if return_attn:
            return [all_candidates, scores], attn
        else:
            return all_candidates, scores


################
# For beam search method

class BeamNode():
    def __init__(self, cur_idx, prob, decoded):
        self.cur_idx = cur_idx
        self.prob = prob
        self.decoded = decoded
        self.is_finished = False

    def __gt__(self, other):
        return self.prob > other.prob

    def __ge__(self, other):
        return self.prob >= other.prob

    def __lt__(self, other):
        return self.prob < other.prob

    def __le__(self, other):
        return self.prob <= other.prob

    def __eq__(self, other):
        return self.prob == other.prob

    def __ne__(self, other):
        return self.prob != other.prob

    def print_spec(self):
        print(f"ID: {self} || cur_idx: {self.cur_idx} || prob: {self.prob} || decoded: {self.decoded}")


class PriorityQueue():
    def __init__(self):
        self.queue = []

    def put(self, obj):
        heapq.heappush(self.queue, (obj.prob, obj))

    def get(self):
        return heapq.heappop(self.queue)[1]

    def qsize(self):
        return len(self.queue)

    def print_scores(self):
        scores = [t[0] for t in self.queue]
        print(scores)

    def print_objs(self):
        objs = [t[1] for t in self.queue]
        print(objs)

