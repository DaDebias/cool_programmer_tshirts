import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd
import json
from numpy import loadtxt
if sys.version_info[0] < 3:
    import io
    open = io.open

model = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin', binary=True) 
#k_model_w2v = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/debiased_model.bin', binary=True) 
#Wl = ['ude', 'hjemme','rig', 'fattig', 'dronning', 'konge', 'skuespillerinde', 'skuespiller']

professions_path = "/work/Exam/cool_programmer_tshirts/data/professions.json"

 with open(professions_file, 'r') as f:
        professions = json.load(f)

Wl = ['revisor', 'sygeplejerske','prinsesse', 'dronning','skuespiller', 'skuespillerinde', 'advokat', 'hjælper', 'ambassadør', 'analytiker', 'antropolog', 'arkæolog', 'ærkebiskop', 'arkitekt', 'kunstner', 'kunstner', 'morder ', 'assistent_professor', 'associate_dean', 'associate_professor', 'astronaut', 'astronom ', 'atlet', 'atletisk_direktør', 'advokat', 'forfatter', 'bager ', 'ballerina', 'boldspiller', 'bankmand', 'barber', 'baron', 'advokat', 'bartender', 'biolog', 'biskop', 'livvagt', 'bogholder', 'chef', 'bokser', 'broadcaster', 'mægler', 'bureaukrat', 'forretningsmand', 'forretningskvinde', 'slagter ', 'butler', 'cab_driver', 'cabbie', 'kameramand', 'kampagneleder', 'kaptajn', 'kardiolog', 'plejer', 'tømrer', 'tegner', 'cellist', 'kansler', 'præst', 'karakter', 'kok', 'kemiker', 'koreograf', 'biograf', 'borger', 'embedsmand', 'præst', 'funktionær', 'træner', 'samler', 'oberst', 'klummeskribent', 'komiker', 'komiker', 'kommandør', 'kommentator', 'kommissær', 'komponist', 'dirigent', 'indrømmer', 'kongresmedlem', 'konstabel', 'konsulent', 'politi', 'korrespondent', 'rådmand', 'rådmand', 'rådgiver', ' kritiker', 'crooner', 'korsfarer', 'kurator', 'formynder', 'far', 'danser', 'dekan', 'tandlæge', 'stedfortræder', 'hudlæge', 'detektiv', 'diplomat', 'instruktør', 'disc_jockey', 'læge', 'doktorand', 'narkoman', 'trommeslager', 'økonomiprofessor', 'økonom', 'redaktør', 'pædagog', 'elektriker', 'medarbejder', 'entertainer', 'iværksætter', 'miljøforkæmper', 'udsending', 'epidemiolog', 'evangelist', 'bonde', 'modedesigner', 'fighter_pilot', 'filmskaber', 'finansmand', 'brandmand', 'brandmand', 'brandmand', 'fisker', 'fodboldspiller', 'foreman', 'freelance_writer', 'gangster', 'gartner', 'geolog', 'målmand', 'grafisk_designer', 'vejleder', 'guitarist', 'frisør', 'handyman', 'skoleleder', 'historiker', 'hitman', 'homemaker', 'hooker', 'husholderske', 'husmor', 'illustratør', 'industriist', 'infielder', 'inspektør', 'instruktør', 'interior_designer', 'opfinder', 'investigator', 'investment_banker', 'pedel', 'juveler', 'journalist', 'dommer', 'jurist', 'arbejder', 'udlejer', 'lovgiver', 'advokat', 'foredragsholder', 'lovgiver', 'bibliotekar', 'løjtnant', 'livredder', 'lyriker', 'maestro', 'magiker', 'magistrat', 'maid', 'major_leaguer', 'manager', 'skytte', 'marshal', 'matematiker', 'mekaniker', 'mediator', 'læge', 'midtbanespiller', 'minister', 'missionær', 'gangster', 'munk', 'musiker', 'barnepige', 'fortæller', 'naturalist', 'forhandler', 'neurolog', 'neurokirurg', 'romanforfatter', 'nunn', 'sygeplejerske', 'observatør', 'officer', 'organist', 'maler', ' advokatfuldmægtig', 'sognebarn', 'parlamentariker', 'præst', 'patolog', 'patruljemand', 'børnlæge', 'performer', 'farmaceut', 'filantrop', 'filosof', 'fotograf', 'fotojournalist', 'læge', 'fysiker', 'pianist', 'planlægger', 'plastikkirurg', 'dramatiker', 'blikkenslager', 'digter', 'politimand', 'politiker', 'pollster', 'prædikant', 'præsident', 'præst', 'rektor', 'fange', 'professor', 'professor_emeritus', 'programmør', 'promotor', 'ejer', 'anklager', 'hovedperson', 'protege', 'demonstrerende', 'prost', 'psykiater', 'psykolog', 'publicist', 'pundit', 'rabbiner', 'radiolog', 'ranger', 'mægler', 'receptionist', 'registreret_sygeplejerske', 'forsker', 'restauratør', 'sømand', 'helgen', 'sælger', 'saxofonist', 'lærd', 'videnskabsmand', 'manuskriptforfatter', 'billedhugger', 'sekretær', 'senator', 'sergent', 'tjener', 'servicemand', 'sheriff_deputy', 'butiksejer', 'sanger', 'singer_songwriter', 'skipper', 'socialite', 'sociolog', 'soft_spoken', 'soldat', 'advokat', 'advokat_general', 'solist', 'sportsmand', 'sportsskriver', 'statsmand', 'steward', 'børsmægler', 'strateg', 'elev', 'stylist', 'substitut', 'superintendent', 'kirurg', 'surveyor', 'svømmer', 'taxichauffør', 'lærer', 'tekniker', 'teenager', 'terapeut', 'handler', 'kasserer', 'trooper', 'trucker', 'trompetist', 'tutor', 'tycoon', 'undersekretær', 'understudy', 'valedictorian', 'vice_chancellor', 'violinist', 'vokalist', 'tjener', 'servitrice', 'vagt', 'kriger', 'svejser', 'arbejder', 'wrestler', 'writer']

y_ax = np.loadtxt('/work/Exam/dk-weat/output/neutral_specific_difference.csv', delimiter=',')
x_ax = ['han', 'hun']

def plot_professions(embedding, wordlist, x_axis, y_axis):

    vectors = []

    words = x_axis + wordlist
    #words =[w for w in words if w in embedding[words]]
    for i in range(len(words)):
        # Embeddings
        vectors.append(embedding[words[i]])
    # To-be basis

    x = (vectors[1]-vectors[0])
    y = y_axis
    
    # Get pseudo-inverse matrix
    W = np.array(vectors)
    B = np.array([x,y])
    Bi = np.linalg.pinv(B.T)
    
    # Project all the words
    Wp = np.matmul(Bi,W.T)
    Wp = (Wp.T-[Wp[0,2],Wp[1,0]]).T
    Wp = Wp.T
    
    df = pd.DataFrame(Wp, index=words, columns=['x', 'y']) # create a dataframe for plotting
    
    # create a plot object
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # add point for the words
    ax.scatter(df['x'], df['y'])
    ax.set_ylabel('genderedness')
    ax.set_xlabel('difference of mand-kvinde')
    ax.axvline(x=0.5)
    # add word label to each point
    for w, pos in df.iterrows():
        ax.annotate(w, pos)

    return ax.plot()

plot_professions(model, Wl[:6],x_ax, y_ax)

