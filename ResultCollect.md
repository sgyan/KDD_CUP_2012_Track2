        # if the loss changes less than 0.001, stops training, does not update model

        # model.load_weights('model_save/deep_fm_fn_bs10000-ep001-loss0.155-val_loss0.153.h5')  # auc: 0.714774
        #model.load_weights('model_save/deep_fm_fn_bs15000-ep001-loss0.156-val_loss0.152.h5')  # auc: 0.717083
        #model.load_weights('model_save/deep_fm_fn-ep002-loss0.154-val_loss0.154-bs15000-ee20-hz[128, 128].h5')  # auc: 0.718581
        #model.load_weights('model_save/deep_fm_fn-ep020-loss0.153-val_loss0.153-bs15000-ee20-hz[5, 600].h5')  # auc: 0.719317
        #model.load_weights('model_save/deep_fm_fn-ep043-loss0.152-val_loss0.152-bs15000-ee20-hz[3, 600].h5')  # auc: 0.722419

        # add dense feature pCTR: over fitting
        # model.load_weights('model_save/deep_fm_combined-ep009-loss0.134-val_loss0.134-bs15000-ee20-hz[3, 600].h5')  # auc: 0.733984
        # model.load_weights('model_save/deep_fm_combined-ep001-loss0.147-val_loss0.139-bs15000-ee20-hz[3, 600].h5')  # auc: 0.744694
        #model.load_weights('model_save/deep_fm_combined-ep003-loss0.135-val_loss0.135-bs15000-ee20-hz[3, 600].h5')  # auc: 0.733826
        #model.load_weights('model_save/deep_fm_combined-ep003-loss0.132-val_loss0.132-bs15000-ee8-hz[3, 600].h5')  # auc: 0.735597
        #model.load_weights('model_save/deep_fm_combined-ep001-loss0.144-val_loss0.135-bs15000-ee8-hz[3, 600].h5')  # auc: 0.743677

        # add pCTR and aCTR
        #model.load_weights('model_save/deep_fm_combined-ep011-loss0.135-val_loss0.135-bs15000-ee20-hz[3, 600].h5')  # auc: 0.738687
        #model.load_weights('model_save/deep_fm_combined-ep001-loss0.150-val_loss0.141-bs15000-ee20-hz[3, 600].h5')  # auc: 0.737412
        #model.load_weights('model_save/deep_fm_combined-ep002-loss0.140-val_loss0.139-bs15000-ee20-hz[3, 600].h5')  # auc: 0.736510

        # add pCTR and group
        #model.load_weights('model_save/deep_fm_combined-ep003-loss0.142-val_loss0.142-bs15000-ee20-hz[3, 600].h5')  # auc: 0.746002
        #model.load_weights('model_save/deep_fm_combined-ep001-loss0.165-val_loss0.146-bs15000-ee20-hz[3, 600].h5')  # auc: 0.739382

        # add pCTR and group and len of tokens
        # model.load_weights('model_save/deep_fm_combined-ep003-loss0.142-val_loss0.142-bs15000-ee20-hz[3, 600].h5')  # auc: 0.748758

        # add pCTR and len of tokens
        # model.load_weights('model_save/deep_fm_combined-ep012-loss0.134-val_loss0.134-bs15000-ee20-hz[3, 600]-t2018-12-17 17:09:17.h5')  # auc: 0.717313
        # model.load_weights('model_save/deep_fm_combined-ep001-loss0.151-val_loss0.141-bs15000-ee20-hz[3, 600]-t2018-12-17 17:09:17.h5')  # auc: 0.739057

        # add len of tokens
        # model.load_weights('model_save/deep_fm_combined-ep013-loss0.153-val_loss0.153-bs15000-ee20-hz[3, 600]-t2018-12-18 00:59:06.h5')  # auc: 0.72

        # add pctr, group, len of token, age, depth, position, rposition
        # model.load_weights('model_save/deep_fm_combined-ep003-loss0.142-val_loss0.142-bs15000-ee20-hz[3, 600]-t2018-12-18 12:48:01.h5')  # auc: 0.749360

        # add pctr, group, len of token, age, depth, position, rposition, num_imp
        # model.load_weights('model_save/deep_fm_combined-ep004-loss0.141-val_loss0.141-bs15000-ee20-hz[3, 600]-t2018-12-18 18:35:59.h5')  # auc: 0.739763

        # add pctr, group, len of token, age, depth, position, rposition, num_occurs
        #model.load_weights('model_save/deep_fm_combined-ep003-loss0.142-val_loss0.142-bs15000-ee20-hz[3, 600]-t2018-12-18 23:26:58.h5')  # auc: 0.750127
        # model.load_weights('model_save/deep_fm_combined-ep003-loss0.141-val_loss0.141-bs18000-ee20-hz[3, 600]-t2018-12-19 11:20:59.h5')  # auc: 0.753482
        # model.load_weights('model_save/deep_fm_combined-ep003-loss0.142-val_loss0.141-bs18000-ee20-hz[3, 500]-t2018-12-19 16:12:38.h5')  # auc: 0.755629
        # model.load_weights('model_save/deep_fm_combined-ep004-loss0.141-val_loss0.141-bs18000-ee20-hz[3, 300]-t2018-12-19 20:46:10.h5')  # auc: 0.751999
        # model.load_weights('model_save/deep_fm_combined-ep005-loss0.140-val_loss0.141-bs18000-ee20-hz[3, 400]-t2018-12-20 10:14:14.h5')  # auc: 0.752596
        # model.load_weights('model_save/deep_fm_combined-ep010-loss0.140-val_loss0.140-bs18000-ee25-hz[3, 500]-t2018-12-20 16:05:34.h5')  # auc: 0.753794

        # add pctr, group, len of token, age, depth, position, rposition, num_occurs, sum_idf
        # model.load_weights('model_save/deep_fm_combined-ep006-loss0.140-val_loss0.140-bs18000-ee20-hz[3, 500]-t2018-12-21 17:17:20.h5')  # auc: 0.753753

        # add pctr, group, len of token, age, depth, position, rposition, num_occurs, tokens_vector
        #model.load_weights('model_save/deep_fm_combined-ep003-loss0.142-val_loss0.141-bs18000-ee20-hz[3, 500]-t2018-12-22 23:35:54.h5')  # auc: 0.761872
        # model.load_weights('model_save/deep_fm_combined-ep007-loss0.140-val_loss0.140-bs18000-ee20-hz[3, 500]-t2018-12-25 18:44:01.h5')  # auc: 0.762638

        # add pctr, group, len of token, age, depth, position, rposition, num_occurs, tokens_vector, id_val
        #model.load_weights('model_save/deep_fm_combined-ep004-loss0.139-val_loss0.139-bs18000-ee8-hz[512, 256, 256, 256, 128]-t2018-12-26 21:32:47.h5')  # auc: 0.769375 778695
        # model.load_weights('model_save/deep_fm_combined-ep005-loss0.139-val_loss0.139-bs18000-ee8-hz[512, 256, 128]-l2l1e-05-l2e1e-05-l2d0-t2018-12-27 13:41:15.h5')  # auc: 0.768785 778826
        #model.load_weights('model_save/deep_fm_combined-ep006-loss0.138-val_loss0.138-bs30000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e1e-05-l2d0.0-t2018-12-27 17:20:56.h5')  # auc: 0.769178 778902
        #model.load_weights('model_save/deep_fm_combined-ep005-loss0.139-val_loss0.139-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e1e-05-l2d1e-05-t2018-12-27 21:42:03.h5')  # auc: 0.769245 780172
        # model.load_weights('model_save/deep_fm_combined-ep005-loss0.139-val_loss0.139-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e1e-05-l2d0.0001-t2018-12-28 08:39:49.h5')  # auc: 0.771686 780086
        # model.load_weights('model_save/deep_fm_combined-ep004-loss0.139-val_loss0.139-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e1e-05-l2d0.001-t2018-12-28 13:39:25.h5')  # auc: 0.769395 780504
        #model.load_weights('model_save/deep_fm_combined-ep003-loss0.141-val_loss0.141-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e0.0001-l2d0.001-t2018-12-28 19:11:03.h5')  # auc: 0.758987 0.770800
        # model.load_weights('model_save/deep_fm_combined-ep002-loss0.134-val_loss0.138-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e0.0-l2d0.001-t2018-12-28 22:40:46.h5')  # auc: 0.759393 0.770754
        #model.load_weights('model_save/deep_fm_combined-ep006-loss0.137-val_loss0.137-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e1e-06-l2d0.001-t2018-12-29 07:15:55.h5')  # auc: 0.775155 0.781973
        #model.load_weights('model_save/deep_fm_combined-ep013-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-29 12:56:43.h5')  # auc: 0.774949 0.783688
        # model.load_weights('model_save/deep_fm_combined-ep009-loss0.117-val_loss0.124-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l0.0-l2e1e-06-l2d0.001-t2018-12-30 00:45:54.h5')  # auc: 0.763515 0.772781
        # model.load_weights('model_save/deep_fm_combined-ep014-loss0.122-val_loss0.123-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-07-l2e1e-06-l2d0.001-t2018-12-30 09:43:53.h5')  # auc: 0.769962 0.778210

        # add group, len of token, age, depth, position, rposition, num_occurs, tokens_vector
        #model.load_weights('model_save/deep_fm_combined-ep004-loss0.150-val_loss0.151-bs18000-ee20-hz[3, 500]-t2018-12-26 10:14:01.h5')  # auc: 0.742817

        # NEW DATA
        # model_name = 'deep_fm_combined-ep013-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 00:27:56.h5'  # auc: 0.773103 0.782441
        # model_name = 'deep_fm_combined-ep009-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 11:40:59.h5'  # auc: 0.774968 0.785623
        # model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 11:40:59.h5'  # auc: 0.774931 0.785993
        # model_name = 'deep_fm_combined-ep005-loss0.129-val_loss0.128-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 11:40:59.h5'  # auc: 0.774381 0.785257
        # new data
        # model_name = 'deep_fm_combined-ep009-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 22:51:48.h5'  # auc: 0.773548 0.783240
        # model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 22:51:48.h5'  # auc: 0.774124 0.784411
        # new data
        # model_name = 'deep_fm_combined-ep009-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-01 11:52:39.h5'  # auc: 0.772808 0.783461
        # model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-01 11:52:39.h5'  # auc: 0.773789 0.784705
        # new data
        # model_name = 'deep_fm_combined-ep011-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-01 20:34:54.h5'  # auc: 0.772387 0.781246
        # model_name = 'deep_fm_combined-ep008-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-01 20:34:54.h5'  # auc: 0.772418 0.782084
        # model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-01 20:34:54.h5'  # auc: 0.772749 0.782517
        # new data
        # model_name = 'deep_fm_combined-ep010-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-02 10:21:56.h5'  # auc: 0.774124 0.785176
        # model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-02 10:21:56.h5'  # auc: 0.775915 0.786094
        # new data
        #model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-02 20:18:42.h5'  # auc: 0.773678 0.784004
        # whole data, stop at epoch 6 base on exp
        # model_name = '.h5'  # auc: 0. 0.

        # add pctr, group, len of token, age, depth, position, rposition, num_occurs, tokens_vector, id_val, sum_mean_idf
#         model_name = 'deep_fm_combined-ep010-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-03 03:45:50.h5'  # auc: 0.774472 0.784256
        #model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-03 03:45:50.h5'  # auc: 0.773730 0.784448
    
        # scoreClickAUC: 0.8210283081506933,0.8479234193047777,0.8632668452084934,0.873112228427833,0.8785250037140251,0.8826570946705343,0.883798759705162
        model_name = 'deep_fm_combined-ep007-loss0.126-val_loss0.126-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-06 12:44:48.h5'  # 0.783213