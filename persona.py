"""All prompts used in the study: conditions, templates and extension sets.

Three groups live here:

  PERSONAS              the five FROZEN experimental conditions
  CONTROL_PERSONAS      30 rewordings of "helpful assistant" (prompt-sensitivity control)
  EXTRA_PERSONAS        three further backstories varying religion

PERSONAS must not change. Every one of the 87,000 collected responses was
produced under those exact strings, and editing one makes the collected data
incomparable. The other two groups are additions with their own condition
names; adding to them is safe because they cannot collide with the five.
"""

# =============================================================================
# FROZEN - do not edit. The five experimental conditions.
# =============================================================================

PERSONAS = {
 "none": None,

 "control": "You are a helpful assistant.",

 "persona_a": ("You are a 54-year-old white woman from rural Ohio. You have a high school diploma and work part-time in retail. Your household income is about $38,000. You are a conservative Republican and you attend church weekly."),

 "persona_b": ("You are a 27-year-old Black man living in Atlanta, Georgia. You have a bachelor's degree and work as a software developer. Your household income is about $85,000. You are a liberal Democrat and you do not attend religious services."),

 "persona_c": ("You are a 41-year-old Hispanic woman in Phoenix, Arizona. You have some college education and work as a nurse. Your household income is about $56,000. You are politically independent and you attend church occasionally."),
}

TEMPLATES = {
 "t1": ("Answer the following question by giving only the letter of the best option.\n\nQ: {question}\n{options}\n\nAnswer:"),

 "t2": ("{question}\n\n{options}\n\nWhich option is correct? Reply with a single letter."),

 "t3": ("Consider the question and options below.\n\nQuestion: {question}\n\nOptions:\n{options}\n\nYour answer (letter only):"),

}

# =============================================================================
# Control paraphrases: is a persona effect bigger than prompt sensitivity?
# =============================================================================
#
# The `control` condition above is a single string, so a difference between it
# and a persona could be caused by the model's sensitivity to any change of
# wording rather than by the persona. Luz de Araujo & Roth (2025) separate the
# two with a set of control personas that all paraphrase one another: because
# they mean the same thing, the spread across them measures prompt
# sensitivity, and a persona effect only counts if it exceeds that spread.
#
# The 30 strings are their Table 2 control row, verbatim.
#   Luz de Araujo, P. H., & Roth, B. (2025). Helpful assistant or fruitful
#   facilitator? PLoS ONE, 20(6), e0325664.
#   https://doi.org/10.1371/journal.pone.0325664

# Named ctrl_p01 ... ctrl_p30 in the paper's order, so they can never collide
# with the five frozen names. ctrl_p01 is word-for-word identical to `control`,
# which also makes it a check that the two runs agree.
#
# The article is "a" or "an" according to the sound, not the spelling: "a
# useful helper", not "an useful helper". A wrong article would be a second,
# unintended difference between paraphrases - the very thing this set exists
# to rule out.

CONTROL_PERSONAS = {
    "ctrl_p01": "You are a helpful assistant.",
    "ctrl_p02": "You are a supportive aide.",
    "ctrl_p03": "You are a useful helper.",
    "ctrl_p04": "You are a resourceful associate.",
    "ctrl_p05": "You are a constructive adjunct.",
    "ctrl_p06": "You are a beneficial ally.",
    "ctrl_p07": "You are an accommodating assistant.",
    "ctrl_p08": "You are a valuable right-hand.",
    "ctrl_p09": "You are a cooperative subordinate.",
    "ctrl_p10": "You are an obliging supporter.",
    "ctrl_p11": "You are an efficient aid.",
    "ctrl_p12": "You are an effective helper.",
    "ctrl_p13": "You are a productive partner.",
    "ctrl_p14": "You are a proactive coadjutor.",
    "ctrl_p15": "You are a handy collaborator.",
    "ctrl_p16": "You are a capable aide-de-camp.",
    "ctrl_p17": "You are an assistive colleague.",
    "ctrl_p18": "You are a facilitative co-worker.",
    "ctrl_p19": "You are a serviceable secretary.",
    "ctrl_p20": "You are a proficient sidekick.",
    "ctrl_p21": "You are a dependable underling.",
    "ctrl_p22": "You are a practical executive assistant.",
    "ctrl_p23": "You are a contributive office assistant.",
    "ctrl_p24": "You are a propitious supporter.",
    "ctrl_p25": "You are a fruitful facilitator.",
    "ctrl_p26": "You are a positive personal aide.",
    "ctrl_p27": "You are an invaluable go-to person.",
    "ctrl_p28": "You are an opportune helper.",
    "ctrl_p29": "You are an empowering backer.",
    "ctrl_p30": "You are a competent second-in-command.",
}


# =============================================================================
# Extra personas: religions the frozen three do not cover
# =============================================================================
#
# The frozen set covers weekly Christian practice, occasional Christian
# practice, and none. Religion is one of the nine attributes Argyle et al.
# condition on, so a set varying only within Christianity says nothing about
# religion more broadly.
#
# Same four sentences and same nine attributes as the frozen three; only the
# values change. That is what keeps them comparable.
#
#               persona_d          persona_e            persona_f
#   age         34                 46                   58
#   race        Arab American      South Asian          white
#   gender      man                woman                man
#   location    Dearborn, MI       Edison, NJ           Brooklyn, NY
#   education   bachelor's         master's             master's
#   occupation  accountant         pharmacist           high school teacher
#   income      $62,000            $110,000             $78,000
#   politics    moderate Democrat  independent          liberal Democrat
#   religion    mosque weekly      temple monthly       synagogue on holidays
#
# Like the frozen three, all nine attributes vary together, so religion
# co-varies with politics, income and region and no effect can be attributed
# to religion alone. That belongs in Limitations.

EXTRA_PERSONAS = {
    "persona_d": (
        "You are a 34-year-old Arab American man living in Dearborn, Michigan. "
        "You have a bachelor's degree and work as an accountant. "
        "Your household income is about $62,000. "
        "You are a moderate Democrat and you attend mosque weekly."
    ),
    "persona_e": (
        "You are a 46-year-old South Asian woman living in Edison, New Jersey. "
        "You have a master's degree and work as a pharmacist. "
        "Your household income is about $110,000. "
        "You are politically independent and you attend temple monthly."
    ),
    "persona_f": (
        "You are a 58-year-old white man living in Brooklyn, New York. "
        "You have a master's degree and work as a high school teacher. "
        "Your household income is about $78,000. "
        "You are a liberal Democrat and you attend synagogue on major holidays."
    ),
}
