import json
import random
import torch
import argparse
import numpy as np

from transformers import pipeline
from transformers import AutoTokenizer, LlamaForCausalLM


def get_args():
    # parsers
    parser = argparse.ArgumentParser(description='llama2')

    parser.add_argument('--data', type=str)

    args = parser.parse_args()
    return args


def generate(generator, instruction, instances, replace_answer, **kwargs):
    for i, instance in enumerate(instances):
        if len(replace_answer) != 0:
            if replace_answer[i] and  not instance['is_impossible']:
                # use the first answer to change
                answer1 = instance['answers'][0]
                context = instance['context'][:answer1['answer_start']] + \
                        replace_answer[i] + \
                        instance['context'][answer1['answer_start']+len(answer1['text']):]
            else:
                context = instance['context']
        prompt = f"[INST] {instruction} \n Context: {context} \n Question: {instance['question']} [/INST]"
        output = generator(prompt,
                        do_sample=kwargs['do_sample'],
                        top_k=kwargs['top_k'],
                        temperature=kwargs['temperature'])[0]['generated_text']
        output = output[len(prompt):].lower() # remove prompt from output
        correct = False
        if instance['is_impossible']:
            answer = "Not possible"
            if answer in output:
                correct = True
        else:
            answer = "\n".join([a['text'] for a in instance['answers']])
            for a in instance['answers']:
                if a['text'] in output:
                    correct = True
                    break
        correct = 'Yes' if correct else 'No'
        print(f'======= prompt {i} =======')
        print(f'=>context: {context}')
        print(f'=>question: {instance["question"]}')
        print(f"=>replace answer: {replace_answer[i]}")
        print(f"=>correct answer:\n {answer}")
        print(f"=>model output:\n {output}")
        print(f'=>model correct?: {correct}')

if __name__=='__main__':
    args = get_args()

    random_seed= 0 
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    #print(open(args.data, "r"))  #it's giving error, arg is NoneType 
    with open(args.data, "r") as f:
        data = json.load(f)
    

    llama2_path = "/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(llama2_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    model = LlamaForCausalLM.from_pretrained(llama2_path, device_map='balanced')
    model.half()

    generator = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_length=2048)

    # modify from here
    generate_args = {
        'do_sample': True,      #to consider more random samples
        'top_k': 30,            #how many words do we chose from
        'temperature': 1.0      #parameter for flattening with softmax function
    }

    print(generate_args)

    instruction = "Given a context, answer the question by extracting answer if you can find it from the context, otherwise answer 'Not possible'."
    
    # define a dictionary: key - index of the instnace, value - num of repetition
    prompt_args = {
        #first_answer = Elie Metchnikoff
        "0":  ["microbiologist",    #similar, semantically close
                "Louis Pasteur",    #dissimilar, semantically close
                "Louis Pasteur",    #synonym
                "Louis Pasteur",             #antonym
                "Quokka",             #unrelated
                "guitar",             #music
                "basketball",             #sports
                "acceleration",             #physics
                "lava",             #geology
                "sauce",            #cuisine
                "microbiology",             #confusion
                "Nobel Prize winner"],               #incorrectness

        #first_answer = a background of the principles of immunology
        "1":    ["immunophysiology",         
                "pharmacology",     
                "Foundations of immunology",     
                "suppressology",   
                "Kung Fu",             
                "guitar",             
                "Dodgeball",             
                "gravity",             
                "sedimentation",             
                "mole",
                "vaccine",
                "magic"],            
        
        #first_answer = enter a special physiological state termed competence
        "2":    ["flow",            
                "be incompetent",  #incompetence           
                "achieve competence",             
                "be incompetent",  #incompetence            
                "use a kaleidoscope",             
                "play a guitar",             
                "play baseball",             
                "have momentum",             
                "undergo erosion",             
                "prepare mole",
                "be competent",
                "confidence"],

        #first_answer = S. pyogenes
        "3":    ["Streptococcus beta-hemolytic",            
                "Streptococcus pneumoniae",             
                "beta-hemolysin-positive staphylococcus",             
                "Non-pyogenic Staphylococcus",             
                "quokka",             
                "guitar",             
                "tennis",             
                "acceleration",             
                "sedimentary",             
                "escargot",
                "Staphylococcus aureus",
                "bacterium"],             
        
        #first_answer=Microbiological
        "4":    ["Microbial",
                "Astronomical",
                "Microbiological", #Microbiology
                "Macrobiological",
                "Artificially intelligent",  #Artificial intelligence
                "Related to drums",  #drums
                "Related to tennis",   #tennis
                "With momentum",
                "Sedimentary",
                "Containing escargot",
                "Microscopic",
                "Bacterial"],
        
        #first_answer=chemotherapy
        "5":    ["antineoplastic therapy",
                 "radiation therapy",
                 "cancer treatment",
                 "hormone therapy",
                 "kiwi",
                 "euphoria",
                 "tennis",
                 "energy",
                 "sedimentary",
                 "exotic",
                 "-",   #i.e. no answer provided on prompt 70
                 "painful"],

        #first_answer=clinical
        "6":    ["medical",
                 "academic",
                 "medical",
                 "non-clinical",
                 "kaleidoscope",
                 "guitar",
                 "bassketball",
                 "acceleration",
                 "Lithosphere",
                 "molecular",
                 "clinical trial",
                 "hospital"],
        
        #first_answer=neuroimmune system
        "7":    ["brain-immune axis",
                 "neuromuscular system",
                 "central nervous system immune system",
                 "Neuroinflammation",
                 "tornado",
                 "guitar",
                 "tennis",
                 "momentum",
                 "Lithosphere",
                 "sauce",
                 "immune system",
                 "neuronome"],
        
        #first_answer=photosynthesis
        "8":    ["chemosynthesis",
                 "decomposition",
                 "plant growth",
                 "respiration",
                 "Astrophysics",
                 "guitar",
                 "baseball",
                 "acceleration",
                 "sediment",
                 "pasta",
                 "photobiosis",
                 "magic"],

        #first_answer=1980s
        "9":    ["Eighties",
                 "dot-com bubble",
                 "the decade of excess",
                 "current",
                 "quokka",
                 "guitar",
                 "basketball",
                 "acceleration",
                 "sedimentary",
                 "mole",
                 "1990s",
                 "Awesome!"],  #prompt 119 is hilarious:)
        
        #first_answer=may only be discovered many years after
        "10":    ["neuroplasticity",
                 "Memeware",
                 "influencer marketing",
                 "must not be discovered many years after",  #may ->must not
                 "Quantumflux",
                 "guitar",
                 "swimming",
                 "orbital",
                 "sedimentary",
                 "sous",
                 "Mullholland",
                 "Zeitgeist"],
        
        #first_answer="80%"
        "11":    ["four-fifths",
                 "40%",
                 "several",
                 "20%",
                 "kangaroo",
                 "guitar",
                 "football",
                 "gravitational",
                 "Igneous",
                 "paella",
                 "blew",
                 "percent"],

        #first_answer="scarring"
        "12":    ["having a fibrosis",
                 "remodeling",
                 "stigmatizing", #stigmatization
                 "healing",
                 "studying astronomy", #astronomy
                 "playing guitar", #guitar
                 "playing baseball", #baseball
                 "having momentum", #momentum
                 "finding Plutonium", #Plutonium
                 "preparing mole",  #mole
                 "scalping",
                 "stretch"],

        #first_answer="commensal flora"
        "13":    ["normal flora",
                 "pathogenic flora",
                 "settling flora", 
                 "pathogenic flora",
                 "Archaea", 
                 "drums", 
                 "fencing", 
                 "gravity", 
                 "sedimentary", 
                 "spices",  
                 "symbiotic flora",
                 "parasite"],
        
        #first_answer="lag phase"
        "14":    ["stationary phase",
                 "exponential growth phase",
                 "stationary phase", 
                 "growth phase",
                 "fermentation", 
                 "cello", 
                 "goalkeeper", 
                 "energy", 
                 "pluton", 
                 "provençal",  
                 "laggard",
                 "delay"],

        #first_answer="infections from being passed from one person to another"
        "15":    ["infections from being transmitted", #transmission
                 "infections from being acquired", #acquisition 
                 "infections from being transmitted", #transmission
                 "immunity", #immunity
                 "fossilization", #fossilization
                 "drums", #drums
                 "goalkeeper", #goalkeeper
                 "quantum", #quantum
                 "sediment", #sediment
                 "escargot",  #escargot
                 "-", #no answer in prompt190
                 "contagion"],

        #first_answer="RNA silencing mechanisms"
        "16":    ["Post-transcriptional gene regulation",
                 "gene editing",
                 "RNA interference",
                 "gene expression",
                 "glycogen",
                 "guitar",
                 "basketball",
                 "relativity",
                 "sedimentary",
                 "mole",
                 "epigenetic silencing",
                 "translation"],
        
        #first_answer="fever and nausea"
        "17":    ["gastrointestinal upset",
                 "insommnia",
                 "vomiting",
                 "hypothermia",
                 "kangaroo",
                 "guitar",
                 "basketball",
                 "momentum",
                 "sedimentation",
                 "tapas",
                 "vomit",
                 "hangover"],
        
        #first_answer="antibiotics do interfere"
        "18":    ["Bacteriostasis",
                 "antivirals",
                 "antibacterials",
                 "probiotics",
                 "floral",
                 "guitar",
                 "baseball",
                 "acceleration",
                 "sedimentary",
                 "mole",
                 "antidepressants",
                 "antibiotics kill"], #kill

        #first_answer="the infectious organism"
        "19":    ["the pathogen",
                 "the machine",
                 "the bacterium",
                 "the non-infectious organism",
                 "pizza",
                 "guitar",
                 "basketball",
                 "acceleration",
                 "cataclysm",
                 "mole",
                 "-", #prompt 238 did not provide answer
                 "germ"],
        
        #first_answer="acid-fast bacilli"
        "20":    ["Mycobacteria",
                "Non-fastidious bacteria",
                "Mycobacterium tuberculosis",
                "fast acid bacilli",
                "caterpillar",
                "guitar",
                "basketball",
                "gravity",
                "sedimentary",
                "mojito",
                "-",   #prompt 250 - no response
                "tuberculosis"],
        
        #first_answer="economic incentives"
        "21":    ["financial incentives",
                "moral incentives",
                "financial incentives",
                "disincentives",
                "nostalgia",
                "guitar",
                "basketball",
                "oscillation",
                "sedimentary",
                "exotic",
                "ecological incentives", 
                "bribe"],
        
        #first_answer="diarrhea"
        "22":    ["-", #no response prompt 264
                "constipation",
                "loose stool or bowel movement",
                "constipation",
                "astronomy",
                "guitar",
                "goal",
                "gravity",
                "sedimentary",
                "molecular",
                "-", 
                "poop"],
            
        #first_answer="Streptococcus"
        "23":    ["Staphylococcus",
                "bacteria",
                "Staphylococcus",
                "Lactobacillus",
                "Nebula",
                "Rhythm",
                "Tennis",
                "Energy",
                "Sedimentary",
                "Sauce",
                "Strep", 
                "Bacteria"],

        #first_answer="Pegylated interferon-alpha-2b"
        "24":    ["pegylated interferon alfa-2b",
                "Omeprazole",
                "PEG-INF",
                "Interferon-alpha-2b (non-pegylated)",
                "Chocolate",
                "Guitar",
                "Tennis",
                "Quantum",
                "Sedimentary",
                "Mise",
                "Pegylated interferon-alpha-2a", 
                "Vaccine"],

        #first_answer="the same species"
        "25":    ["primate", #misunderstood to human
                "Feral",
                "-", #gave incorrect answer
                "domain",
                "kangaroo",
                "guitar",
                "dodgeball",
                "acceleration",
                "sedimentary",
                "sous",
                "chimpanzee", 
                "doggo"],

        #first_answer="a host organism"
        "26":    ["vectors",
                "parasite",
                "vector",
                "parasite",
                "Chlamydosaurus",
                "guitar",
                "basketball",
                "acceleration",
                "igneous",
                "Moussaka",
                "parasite", 
                "guest"],

        #first_answer="a measure of protection"
        "27":    ["safeguard",
                "risk",
                "safeguards",
                "vulnerability",
                "chocolate",
                "guitar",
                "tennis",
                "acceleration",
                "basalt",
                "molecular",
                "moisture", 
                "insurance"],

        #first_answer="cellular response"
        "28":    ["cellular reaction",
                "molecular response",
                "Cellular reaction",
                "tissue response",
                "caterpillar",
                "guitar",
                "dodgeball",
                "energy",
                "volcano",
                "tapas",
                "celestial", 
                "fight"],

        #first_answer=""
        "29":    ["traditional medicine",
                "scientific evidence",
                "herbalism",
                "scientific evidence",
                "space exploration",
                "guitar",
                "tennis",
                "quantum",
                "Pluton",
                "mole",
                "mandrake", 
                "myth"]

          # the length of this dictionary is going to be the num_instances var in the test_prompt.sh
    }

    instances = []
    replace_answers = []
    for i in range(len(prompt_args)):
        repeat_time = len(prompt_args[str(i)])
        replace_answer = prompt_args[str(i)]
        instance_i = [data[i]] * repeat_time
        instances += instance_i
        replace_answers += replace_answer
        assert len(instances) == len(replace_answers)

    generate(generator, instruction, instances, replace_answers, **generate_args)

