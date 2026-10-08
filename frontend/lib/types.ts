export type Repository = {
  id: string;
  owner: string;
  name: string;
  full_name: string;
  provider?: string;
  api_base_url?: string;
  clone_url?: string | null;
  local_path?: string | null;
  checkout_mode?: string;
  description?: string;
  default_branch?: string;
  last_sync_at?: string | null;
};
