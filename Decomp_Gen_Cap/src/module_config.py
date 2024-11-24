import argparse 
"""
configurations for module implementation
    e.g., thresholds for left/right relation check
    what coordinates should be considered as on the left hand side, etc.
"""
def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument('--DEBUG', 
                        type=bool,
                        default=False)
    
    #model configurations
    parser.add_argument('--OBJDET_MODEL_NAME', 
                        type=str,
                        default="google/owlvit-large-patch14")
    parser.add_argument('--VQA_MODEL_NAME', 
                        type=str,
                        default="Salesforce/blip2-flan-t5-xl")
    parser.add_argument('--TEXT_MODEL_NAME', 
                        type=str,
                        default="google/owlvit-large-patch14")
    parser.add_argument('--GRAMMAR_MODEL_NAME', 
                        type=str,
                        default="textattack/distilbert-base-uncased-CoLA")

    #special term configurations
    parser.add_argument('--SPLIT_TERM', 
                        type=str,
                        default="###")
    parser.add_argument('--CHUNK_LENGTH', 
                        type=str,
                        default=8)
    parser.add_argument('--CHUNK_LENGTH_QUES', 
                        type=str,
                        default=8)
    parser.add_argument('--MAX_QUES_LENGTH', 
                        type=str,
                        default=30)

    #threshold configurations
    parser.add_argument('--BBOX_CONFIDENCE', 
                        type=float,
                        default=0.25)
    #this value is set with empirical analysis ==> use the debug mode to see the valid scores
    parser.add_argument('--VALID_QUES_SCORE', 
                        type=float,
                        default=0.75)
    parser.add_argument('--LARGE_WIDTH', 
                        type=float,
                        default=0.4)
    parser.add_argument('--LARGE_HEIGHT', 
                        type=float,
                        default=0.4)
    parser.add_argument('--SMALL_WIDTH', 
                        type=float,
                        default=0.3)
    parser.add_argument('--SMALL_HEIGHT', 
                        type=float,
                        default=0.3)
    parser.add_argument('--LONG_SIZE', 
                        type=float,
                        default=0.5)
    parser.add_argument('--SHORT_SIZE', 
                        type=float,
                        default=0.3)
    parser.add_argument('--HEIGHT_SIZE', 
                        type=float,
                        default=0.4)
    parser.add_argument('--DIFF_THRESHOLD', 
                        type=float,
                        default=0.1)
    args=parser.parse_args()
    return args