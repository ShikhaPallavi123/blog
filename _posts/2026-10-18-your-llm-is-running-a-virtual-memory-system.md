---
layout: post
title: "Your LLM Is Running a Virtual Memory System"
---
In my simulation, replacing "reserve the maximum length" with paged allocation lets 2.1x more LLM requests fit in the same GPU memory, an idea borrowed straight from 1960s operating systems. Then I tried to beat the classic cache-eviction policy for AI agents, and my first two attempts lost to plain LRU by up to 40 points.

## The operating-systems idea in five minutes

This comes from chapters 18-22 of *Operating Systems: Three Easy Pieces* (Arpaci-Dusseau and Arpaci-Dusseau).

A process wants one big continuous block of memory, but physical memory is limited and gets chopped up as programs start and stop. **Paging** solves this by cutting memory into fixed-size pages. Any page can sit anywhere, and a **page table** maps what the program sees to where the data really is. Fixed sizes remove external fragmentation. The price is a little waste inside the last, partly filled page.

When memory is full, the OS must choose a victim page to evict. The book walks through the policies: FIFO, random, LRU (evict the page unused for the longest), and Belady's optimal policy, which evicts the page needed furthest in the future. Optimal cannot be built because it needs to see the future, but it is the ruler every real policy is measured against.

The book's most useful warning is the **looping-sequential workload**: 50 pages accessed in order, over and over, with room for only 49. LRU and FIFO evict exactly the page about to be needed, so they score a 0% hit rate. Random, the book notes, does notably better. I reproduced this: over 200 loops, FIFO and LRU hit 0.0% and random hit 95.4%.

Keep that failure in mind. Agents loop.

## Where it reappears in AI

When an LLM generates text, it stores a key and a value vector for every token it has seen, at every layer, so it does not recompute them for each new token. This is the **KV cache**, and it often dominates GPU memory in serving. Its size depends on how long the output turns out to be, which nobody knows in advance.

Early serving systems handled this the way early operating systems handled processes: reserve the worst case, contiguously, up front. Most of that reserved space sat empty. The vLLM paper (Kwon et al., SOSP 2023) fixed it with **PagedAttention**: split the KV cache into small fixed-size blocks and keep a **block table** per request. That is a page table in everything but name, and it removes most of the waste.

The same ideas now run one level up. **Prefix caching** (vLLM's automatic prefix caching, SGLang's RadixAttention) keeps the KV blocks of prompts it has already seen, so the next request that starts with the same text skips the work. That is a page cache for prompts, and like any cache it needs a replacement policy.

Agents make the policy matter. An agent loop re-sends its whole growing history at every step, waits seconds for a tool to return, then comes back. Each session is a process that sleeps for an unpredictable time and then re-reads everything. The question from chapter 22 returns in a new form: **which sleeping session's memory do you evict?**

## Experiment 1: reserve-the-maximum vs paged blocks

**This is a simulation, not a benchmark of vLLM.** I generated 20,000 synthetic requests: prompt length lognormal with median 500 tokens, output length lognormal with median 200 tokens, maximum sequence 2,048 tokens. The static allocator reserves 2,048 slots per request. The paged allocator uses 16-token blocks, allocated as the request grows. Code: [os-meets-ai](https://github.com/ShikhaPallavi123/os-meets-ai).

| Allocator | Memory holding live tokens | Requests that fit in 200,000 token slots |
| --- | --- | --- |
| Reserve maximum length | 46.4% | 97.7 |
| Paged, 16-token blocks | 99.3% | 209.1 |

The average request ends with 950 live tokens, so reserving 2,048 wastes more than half. Paging wastes at most one partly filled block per request. Result: **2.14x more concurrent requests** in the same memory. The exact ratio depends on your length distribution and maximum, which is why the claim to take away is the mechanism, not the number.

A replication on a real GPU with vLLM, using a real length distribution, is the next step and will be a follow-up post.

## Experiment 2: which sleeping agent do you evict?

**Also a simulation.** 60 agent sessions. Each starts with a 64-block prompt. At every step the session re-reads its entire cached prefix, appends new blocks (median 24, lognormal), then waits for a tool. Tool latency is exponential with a mean of 1, 5 or 30 seconds, fixed per session. Each session runs 10-20 steps. Eviction removes tail blocks, as radix-tree prefix caches do, so a session's cached prefix stays contiguous. Cache size is a percentage of the peak working set. Results are the mean of 5 seeds; the metric is the share of previously computed prefix blocks served from cache.

![Prefix-cache hit rate by eviction policy on synthetic agent traces]({{ "/assets/figure1_prefix_cache_policies.png" | relative_url }})

I tried three ideas against LRU, random, and an oracle that knows when each session will return.

1. **Predicted return (v1):** evict the session expected back last (last access plus its average gap). At a cache of 20% of the working set it scored 19.9%, against 43.2% for LRU.
2. **Return-rate-aware (v2):** evict the session that returns least often. It scored 27.3% at 20%, and 37.9% against LRU's 78.5% at 50%.
3. **Rate plus idle time (v3):** use the larger of the usual gap and how long the session has already been idle. It scored 44.8% at 20% and 76.4% at 50%, roughly tied with LRU.

My explanation for v1 and v2 failing is that they keep finished sessions cached forever, because a finished session looks like a fast one that is merely late. v3 recovers once idle time counts, which fits that story, but I did not measure how many dead sessions each policy held, so treat it as a hypothesis.

The oracle is the real finding. At a cache of 20%, it reaches 66.7% against LRU's 43.2%, so a better predictor has about 23 points of hit rate to win. LRU leaves that on the table, and simple frequency-and-idle heuristics did not claim it.

I also tuned v1 to v3 on the same synthetic generator I report on, so the comparison among my own policies is optimistic. Only the ranking against LRU and the oracle gap should be taken from this.

## What is missing

- **Real traces.** My latencies are memoryless and my sessions are independent. Real agents have retries, tool-specific latency tails, and many sessions sharing the same system prompt, which a radix tree exploits and my model ignores.
- **Feedback.** A cache miss slows a session down and shifts its next arrival. I fixed the arrival times up front, so that loop is missing.
- **Cost-aware eviction.** A miss costs recompute time proportional to the prefix length, so a long context is more expensive to lose than a short one. Policies from the web-cache literature that weigh size and cost are the natural thing to try.
- **Tiers.** Chapter 21 covers swap space. The real choice is not drop or keep but GPU, CPU memory or SSD, each with its own reload cost.
- **Session-end signals.** Agent frameworks know when a task finishes. Serving engines mostly do not.

## What I would build next

**Hypothesis (untested):** if the agent framework passes hints to the cache, namely "session finished", an expected tool-latency class and a remaining step budget, then evicting by expected reuse times recompute cost beats LRU on real agent traces.

**How it dies:** replay real traces through a patched eviction policy in vLLM or SGLang. If hit rate and p99 time-to-first-token do not improve over LRU, the idea is wrong and I will say so.

**Resources needed:** one 24-80 GB GPU for a few days; real agent traces with tool timings (I will log my own runs, and would welcome anonymized traces from a team that can share them); and a maintainer willing to review a patch. If you work on LLM serving and want to collaborate, email me.

Next in the series: agents are processes, so what is their kernel?

## References

- Arpaci-Dusseau, R. and Arpaci-Dusseau, A. *Operating Systems: Three Easy Pieces*, chapters 18-22.
- Belady, L. A. "A Study of Replacement Algorithms for a Virtual-Storage Computer." IBM Systems Journal, 1966.
- Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
- Zheng, L. et al. "SGLang: Efficient Execution of Structured Language Model Programs" (RadixAttention), 2023.
- Packer, C. et al. "MemGPT: Towards LLMs as Operating Systems," 2023.
