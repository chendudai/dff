#import plotly.graph_objects as go
#import numpy as np
#
#import plotly.express as px
#
#df = px.data.gapminder().query("country=='Canada'")
#fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
#fig.show()
#fig.write_html("plotly example")
## %%
import pandas as pd
import matplotlib
#
## each experiment is saved to a metrics.csv file which can be imported anywhere
## images save to exp/version/images
#df = pd.read_csv('./out2/logs/exp_HaNeRF_reg/version_58/metrics.csv')
#df['train/loss'].plot(title="train\loss - feature loss weights (coarse and fine) 0.04")
#matplotlib.pyplot.figure()
#df['train/features_coarse'].plot(title="train\features_coarse loss")
#matplotlib.pyplot.figure()
#df['train/features_fine'].plot(title="train\features_fine loss")

##import plotly.express as px
##
##fig =px.scatter(x=range(10), y=range(10))
##fig.write_html("path/to/file.html")
# %%
import clip_utils
import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2
import pandas
#
##def get_k_files(k, csv_path, prompt):
##    xls_file = pandas.read_csv(csv_path)
##    col = xls_file[prompt]
##    col_sorted = col.sort_values(by=prompt, ascending=False)
##    files = col_sorted[:k]
##    names = xls_file['filename'][files.index]
##    return names.values.tolist()
##get_k_files(k, "", "a picture of a cathedral's window")
clip_editor = clip_utils.CLIPEditor()
#clip_str = ["apple","banana","vegetable","floor"]
clip_str = ["a_picture_of_a_cathedral's_window"]
clip_editor.text_features = clip_editor.encode_text([t.replace('_', ' ') for t in clip_str])
print([t.replace('_', ' ') for t in clip_str])
print(clip_editor.text_features @ clip_editor.text_features.T)
print("shape",clip_editor.text_features.shape)

feature_map = torch.load("/storage/hanibezalel/distilled-feature-fields/data/chen_dataset/rgb_feature_langseg/0000_fmap_CxHxW.pt")[None].float()
feature_map = feature_map.squeeze().view(512, -1).permute(1, 0).to("cuda")
scores = clip_editor.calculate_selection_score(feature_map, query_features=clip_editor.text_features)
score_patch = scores.reshape(1, 360, 480, 1).permute(0, 3, 1, 2).detach()
score_pred = (score_patch.detach().permute(0, 2, 3, 1)[0].cpu().numpy()*255).astype(np.uint8)[:, :, 0]
#imageio.imsave("out/test_score.png",score_pred)
activation_heatmap = cv2.applyColorMap(score_pred, cv2.COLORMAP_JET)
cv2.imwrite("out/test_score.png", activation_heatmap)
#
# %%
