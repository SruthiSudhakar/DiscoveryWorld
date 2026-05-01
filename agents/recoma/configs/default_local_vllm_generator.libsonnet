// Generator config for a local vLLM-served, OpenAI-compatible endpoint.
// MODEL must use the `openai/<hf_model_id>` form (e.g. `openai/Qwen/Qwen3-32B`)
// so LiteLLM routes to its OpenAI client, which then picks up
// OPENAI_BASE_URL + OPENAI_API_KEY from the environment (set in
// run/run_react_easy_oss.sh) and talks to the local vLLM server.
//
// recoma's `lite_llm` only forwards a fixed set of generator params
// (temperature, max_tokens, top_p, n, stop, seed, etc.). litellm.drop_params=True
// is set inside recoma, so any params vLLM does not understand are silently
// dropped — safe to keep `seed` here.
{
    "type": "lite_llm",
    "model": std.extVar("MODEL"),
    "max_tokens": 100,
    "temperature": 0.0,
    "use_cache": true,
    "seed": std.parseInt(std.extVar("SEED")),
    "stop": ["\n"],
}
