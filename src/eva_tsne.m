function eva_tsne(emb, label, dmnetype)

addpath('tsne');

if strcmp(dmnetype, 'ed')
    titlename = 'DMNE (ED)';
else
    titlename = 'DMNE (PD)';
end

figure;
colormap jet;
map_emb = tsne(emb, label, 2, 30, 30);

set(gca,'xtick',[]);
set(gca,'ytick',[]);
set(gca,'xticklabel',[]);
set(gca,'yticklabel',[]);
box on;

title(titlename, 'fontname', 'Arial', 'fontsize', 25);

end
