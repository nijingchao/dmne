%% write embeddings

function write_emb(embs, destdir, dmnetype)

numnet = length(embs);

for i = 1:numnet
    emb_i = embs{i};
    numnode = size(emb_i, 1);
    numdim = size(emb_i, 2);
    destpath = [destdir 'emb_' dmnetype '_' num2str(i) '.txt'];
    fid = fopen(destpath, 'w');
    fprintf(fid, '%d, %d\n', numnode, numdim);
    for j = 1:numnode
        fprintf(fid, '%d ', j);
        for k = 1:numdim
            if k < numdim
                fprintf(fid, '%.6f ', emb_i(j, k));
            else
                fprintf(fid, '%.6f\n', emb_i(j, k));
            end
        end
    end
    fclose(fid);
end

end
